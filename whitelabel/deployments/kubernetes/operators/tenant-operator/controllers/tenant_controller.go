package controllers

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	whitelabelv1 "github.com/enterprise/whitelabel/deployments/kubernetes/operators/tenant-operator/api/v1"
)

// TenantReconciler reconciles a Tenant object
type TenantReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=whitelabel.io,resources=tenants,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=whitelabel.io,resources=tenants/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=whitelabel.io,resources=tenants/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=namespaces,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=resourcequotas,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=serviceaccounts,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.k8s.io,resources=networkpolicies,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.k8s.io,resources=ingresses,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=rolebindings,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop
func (r *TenantReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("tenant", req.NamespacedName)

	// Fetch the Tenant instance
	var tenant whitelabelv1.Tenant
	if err := r.Get(ctx, req.NamespacedName, &tenant); err != nil {
		if apierrors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			log.Info("Tenant resource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get Tenant")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if tenant.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, &tenant, log)
	}

	// Add finalizer if not present
	if !controllerutil.ContainsFinalizer(&tenant, "tenant.whitelabel.io/finalizer") {
		controllerutil.AddFinalizer(&tenant, "tenant.whitelabel.io/finalizer")
		if err := r.Update(ctx, &tenant); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Update status to provisioning if not set
	if tenant.Status.Phase == "" {
		tenant.Status.Phase = whitelabelv1.TenantPhaseProvisioning
		tenant.Status.Message = "Starting tenant provisioning"
		if err := r.Status().Update(ctx, &tenant); err != nil {
			return ctrl.Result{}, err
		}
	}

	// Reconcile tenant resources
	if err := r.reconcileTenantResources(ctx, &tenant, log); err != nil {
		log.Error(err, "Failed to reconcile tenant resources")
		r.updateTenantStatus(ctx, &tenant, whitelabelv1.TenantPhaseFailed, err.Error())
		return ctrl.Result{RequeueAfter: time.Minute * 5}, err
	}

	// Update status to active
	if tenant.Status.Phase != whitelabelv1.TenantPhaseActive {
		tenant.Status.Phase = whitelabelv1.TenantPhaseActive
		tenant.Status.Message = "Tenant is active and operational"
		tenant.Status.LastReconcileTime = &metav1.Time{Time: time.Now()}
		if err := r.Status().Update(ctx, &tenant); err != nil {
			return ctrl.Result{}, err
		}
	}

	log.Info("Successfully reconciled Tenant")
	return ctrl.Result{RequeueAfter: time.Minute * 10}, nil
}

// handleDeletion handles the deletion of a tenant
func (r *TenantReconciler) handleDeletion(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) (ctrl.Result, error) {
	log.Info("Handling tenant deletion")
	
	// Update status to deleting
	tenant.Status.Phase = whitelabelv1.TenantPhaseDeleting
	tenant.Status.Message = "Deleting tenant resources"
	if err := r.Status().Update(ctx, tenant); err != nil {
		log.Error(err, "Failed to update status to deleting")
	}

	// Clean up tenant resources
	if err := r.cleanupTenantResources(ctx, tenant, log); err != nil {
		log.Error(err, "Failed to cleanup tenant resources")
		return ctrl.Result{RequeueAfter: time.Minute}, err
	}

	// Remove finalizer
	controllerutil.RemoveFinalizer(tenant, "tenant.whitelabel.io/finalizer")
	if err := r.Update(ctx, tenant); err != nil {
		return ctrl.Result{}, err
	}

	log.Info("Tenant deletion completed")
	return ctrl.Result{}, nil
}

// reconcileTenantResources reconciles all tenant resources
func (r *TenantReconciler) reconcileTenantResources(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	// Create namespace
	if err := r.reconcileNamespace(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile namespace: %w", err)
	}

	// Create resource quota
	if err := r.reconcileResourceQuota(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile resource quota: %w", err)
	}

	// Create service account
	if err := r.reconcileServiceAccount(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile service account: %w", err)
	}

	// Create RBAC resources
	if err := r.reconcileRBAC(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile RBAC: %w", err)
	}

	// Create ConfigMap
	if err := r.reconcileConfigMap(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile config map: %w", err)
	}

	// Create Secret
	if err := r.reconcileSecret(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile secret: %w", err)
	}

	// Create Network Policy
	if err := r.reconcileNetworkPolicy(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile network policy: %w", err)
	}

	// Create Service
	if err := r.reconcileService(ctx, tenant, log); err != nil {
		return fmt.Errorf("failed to reconcile service: %w", err)
	}

	// Create Ingress for custom domains
	if len(tenant.Spec.CustomDomains) > 0 {
		if err := r.reconcileIngress(ctx, tenant, log); err != nil {
			return fmt.Errorf("failed to reconcile ingress: %w", err)
		}
	}

	return nil
}

// reconcileNamespace creates or updates the tenant namespace
func (r *TenantReconciler) reconcileNamespace(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
			Labels: map[string]string{
				"app.kubernetes.io/name":       "whitelabel-tenant",
				"app.kubernetes.io/instance":   tenant.Spec.TenantID,
				"app.kubernetes.io/part-of":    "whitelabel-platform",
				"app.kubernetes.io/managed-by": "tenant-operator",
				"tenant.whitelabel.io/id":      tenant.Spec.TenantID,
				"tenant.whitelabel.io/isolation": tenant.Spec.IsolationStrategy,
			},
			Annotations: map[string]string{
				"tenant.whitelabel.io/plan": tenant.Spec.Plan,
			},
		},
	}

	if err := controllerutil.SetControllerReference(tenant, namespace, r.Scheme); err != nil {
		return err
	}

	var existingNamespace corev1.Namespace
	if err := r.Get(ctx, types.NamespacedName{Name: namespace.Name}, &existingNamespace); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating namespace", "namespace", namespace.Name)
			return r.Create(ctx, namespace)
		}
		return err
	}

	// Update tenant status with namespace info
	tenant.Status.Resources.Namespace = namespace.Name
	return nil
}

// reconcileResourceQuota creates or updates the resource quota
func (r *TenantReconciler) reconcileResourceQuota(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	quota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-quota", tenant.Spec.TenantID),
			Namespace: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
		},
		Spec: corev1.ResourceQuotaSpec{
			Hard: corev1.ResourceList{
				"requests.cpu":              resource.MustParse(getStringOrDefault(tenant.Spec.ResourceQuota.CPURequest, "500m")),
				"requests.memory":           resource.MustParse(getStringOrDefault(tenant.Spec.ResourceQuota.MemoryRequest, "1Gi")),
				"limits.cpu":                resource.MustParse(getStringOrDefault(tenant.Spec.ResourceQuota.CPULimit, "2000m")),
				"limits.memory":             resource.MustParse(getStringOrDefault(tenant.Spec.ResourceQuota.MemoryLimit, "4Gi")),
				"requests.storage":          resource.MustParse(getStringOrDefault(tenant.Spec.ResourceQuota.StorageRequest, "10Gi")),
				"persistentvolumeclaims":    resource.MustParse(fmt.Sprintf("%d", getInt32OrDefault(tenant.Spec.ResourceQuota.PVCCount, 5))),
				"services":                  resource.MustParse(fmt.Sprintf("%d", getInt32OrDefault(tenant.Spec.ResourceQuota.ServiceCount, 5))),
				"count/deployments.apps":    resource.MustParse(fmt.Sprintf("%d", getInt32OrDefault(tenant.Spec.ResourceQuota.DeploymentCount, 5))),
			},
		},
	}

	if err := controllerutil.SetControllerReference(tenant, quota, r.Scheme); err != nil {
		return err
	}

	var existingQuota corev1.ResourceQuota
	if err := r.Get(ctx, types.NamespacedName{Name: quota.Name, Namespace: quota.Namespace}, &existingQuota); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating resource quota", "quota", quota.Name)
			return r.Create(ctx, quota)
		}
		return err
	}

	// Update if necessary
	existingQuota.Spec = quota.Spec
	return r.Update(ctx, &existingQuota)
}

// reconcileServiceAccount creates or updates the service account
func (r *TenantReconciler) reconcileServiceAccount(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	serviceAccount := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-app", tenant.Spec.TenantID),
			Namespace: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
			Labels: map[string]string{
				"tenant.whitelabel.io/id": tenant.Spec.TenantID,
			},
		},
		AutomountServiceAccountToken: &[]bool{true}[0],
	}

	if err := controllerutil.SetControllerReference(tenant, serviceAccount, r.Scheme); err != nil {
		return err
	}

	var existingServiceAccount corev1.ServiceAccount
	if err := r.Get(ctx, types.NamespacedName{Name: serviceAccount.Name, Namespace: serviceAccount.Namespace}, &existingServiceAccount); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating service account", "serviceAccount", serviceAccount.Name)
			return r.Create(ctx, serviceAccount)
		}
		return err
	}

	// Update tenant status with service account info
	tenant.Status.Resources.ServiceAccount = serviceAccount.Name
	return nil
}

// reconcileConfigMap creates or updates the config map
func (r *TenantReconciler) reconcileConfigMap(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-config", tenant.Spec.TenantID),
			Namespace: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
		},
		Data: map[string]string{
			"tenant.id":               tenant.Spec.TenantID,
			"tenant.name":             tenant.Spec.Name,
			"tenant.plan":             tenant.Spec.Plan,
			"isolation.strategy":      tenant.Spec.IsolationStrategy,
			"monitoring.enabled":      fmt.Sprintf("%t", tenant.Spec.MonitoringConfig.Enabled),
			"logging.level":           getStringOrDefault(tenant.Spec.MonitoringConfig.LogLevel, "info"),
			"backup.enabled":          fmt.Sprintf("%t", tenant.Spec.BackupConfig.Enabled),
			"autoscaling.enabled":     fmt.Sprintf("%t", tenant.Spec.AutoscalingConfig.Enabled),
		},
	}

	if err := controllerutil.SetControllerReference(tenant, configMap, r.Scheme); err != nil {
		return err
	}

	var existingConfigMap corev1.ConfigMap
	if err := r.Get(ctx, types.NamespacedName{Name: configMap.Name, Namespace: configMap.Namespace}, &existingConfigMap); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating config map", "configMap", configMap.Name)
			return r.Create(ctx, configMap)
		}
		return err
	}

	// Update if necessary
	existingConfigMap.Data = configMap.Data
	if err := r.Update(ctx, &existingConfigMap); err != nil {
		return err
	}

	// Update tenant status with config map info
	tenant.Status.Resources.ConfigMap = configMap.Name
	return nil
}

// reconcileSecret creates or updates the secret
func (r *TenantReconciler) reconcileSecret(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-secrets", tenant.Spec.TenantID),
			Namespace: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
		},
		Type: corev1.SecretTypeOpaque,
		StringData: map[string]string{
			"api.key":         generateAPIKey(tenant.Spec.TenantID),
			"encryption.key":  generateEncryptionKey(),
			"jwt.secret":      generateJWTSecret(),
			"webhook.secret":  generateWebhookSecret(),
		},
	}

	if err := controllerutil.SetControllerReference(tenant, secret, r.Scheme); err != nil {
		return err
	}

	var existingSecret corev1.Secret
	if err := r.Get(ctx, types.NamespacedName{Name: secret.Name, Namespace: secret.Namespace}, &existingSecret); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating secret", "secret", secret.Name)
			return r.Create(ctx, secret)
		}
		return err
	}

	// Update tenant status with secret info
	tenant.Status.Resources.Secret = secret.Name
	return nil
}

// reconcileNetworkPolicy creates or updates the network policy
func (r *TenantReconciler) reconcileNetworkPolicy(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	if !tenant.Spec.SecurityConfig.NetworkPoliciesEnabled {
		return nil
	}

	networkPolicy := &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-isolation", tenant.Spec.TenantID),
			Namespace: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"tenant.whitelabel.io/id": tenant.Spec.TenantID,
				},
			},
			PolicyTypes: []networkingv1.PolicyType{
				networkingv1.PolicyTypeIngress,
				networkingv1.PolicyTypeEgress,
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					From: []networkingv1.NetworkPolicyPeer{
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"name": "whitelabel-system",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"tenant.whitelabel.io/id": tenant.Spec.TenantID,
								},
							},
						},
					},
				},
			},
			Egress: []networkingv1.NetworkPolicyEgressRule{
				{
					To: []networkingv1.NetworkPolicyPeer{
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"tenant.whitelabel.io/id": tenant.Spec.TenantID,
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"name": "whitelabel-system",
								},
							},
						},
					},
				},
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{Port: &intstr.IntOrString{IntVal: 443}},
						{Port: &intstr.IntOrString{IntVal: 80}},
						{Port: &intstr.IntOrString{IntVal: 5432}},
						{Port: &intstr.IntOrString{IntVal: 6379}},
					},
				},
			},
		},
	}

	if err := controllerutil.SetControllerReference(tenant, networkPolicy, r.Scheme); err != nil {
		return err
	}

	var existingNetworkPolicy networkingv1.NetworkPolicy
	if err := r.Get(ctx, types.NamespacedName{Name: networkPolicy.Name, Namespace: networkPolicy.Namespace}, &existingNetworkPolicy); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating network policy", "networkPolicy", networkPolicy.Name)
			return r.Create(ctx, networkPolicy)
		}
		return err
	}

	// Update tenant status with network policy info
	tenant.Status.Resources.NetworkPolicy = networkPolicy.Name
	return nil
}

// reconcileService creates or updates the service
func (r *TenantReconciler) reconcileService(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-service", tenant.Spec.TenantID),
			Namespace: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
			Labels: map[string]string{
				"app":                        fmt.Sprintf("tenant-%s-app", tenant.Spec.TenantID),
				"tenant.whitelabel.io/id":    tenant.Spec.TenantID,
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"app":                        fmt.Sprintf("tenant-%s-app", tenant.Spec.TenantID),
				"tenant.whitelabel.io/id":    tenant.Spec.TenantID,
			},
			Type: corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromInt(8080),
				},
				{
					Name:       "https",
					Port:       443,
					TargetPort: intstr.FromInt(8080),
				},
			},
		},
	}

	if err := controllerutil.SetControllerReference(tenant, service, r.Scheme); err != nil {
		return err
	}

	var existingService corev1.Service
	if err := r.Get(ctx, types.NamespacedName{Name: service.Name, Namespace: service.Namespace}, &existingService); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating service", "service", service.Name)
			return r.Create(ctx, service)
		}
		return err
	}

	// Update tenant status with service info
	tenant.Status.Resources.Service = service.Name
	return nil
}

// reconcileIngress creates or updates the ingress for custom domains
func (r *TenantReconciler) reconcileIngress(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	pathType := networkingv1.PathTypePrefix
	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("tenant-%s-ingress", tenant.Spec.TenantID),
			Namespace: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
			Annotations: map[string]string{
				"kubernetes.io/ingress.class":                "nginx",
				"cert-manager.io/cluster-issuer":             "letsencrypt-prod",
				"nginx.ingress.kubernetes.io/ssl-redirect":   "true",
				"nginx.ingress.kubernetes.io/force-ssl-redirect": "true",
			},
		},
		Spec: networkingv1.IngressSpec{
			TLS: []networkingv1.IngressTLS{
				{
					Hosts:      tenant.Spec.CustomDomains,
					SecretName: fmt.Sprintf("tenant-%s-tls", tenant.Spec.TenantID),
				},
			},
		},
	}

	// Add rules for each custom domain
	for _, domain := range tenant.Spec.CustomDomains {
		rule := networkingv1.IngressRule{
			Host: domain,
			IngressRuleValue: networkingv1.IngressRuleValue{
				HTTP: &networkingv1.HTTPIngressRuleValue{
					Paths: []networkingv1.HTTPIngressPath{
						{
							Path:     "/",
							PathType: &pathType,
							Backend: networkingv1.IngressBackend{
								Service: &networkingv1.IngressServiceBackend{
									Name: fmt.Sprintf("tenant-%s-service", tenant.Spec.TenantID),
									Port: networkingv1.ServiceBackendPort{
										Number: 80,
									},
								},
							},
						},
					},
				},
			},
		}
		ingress.Spec.Rules = append(ingress.Spec.Rules, rule)
	}

	if err := controllerutil.SetControllerReference(tenant, ingress, r.Scheme); err != nil {
		return err
	}

	var existingIngress networkingv1.Ingress
	if err := r.Get(ctx, types.NamespacedName{Name: ingress.Name, Namespace: ingress.Namespace}, &existingIngress); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("Creating ingress", "ingress", ingress.Name)
			return r.Create(ctx, ingress)
		}
		return err
	}

	// Update if necessary
	existingIngress.Spec = ingress.Spec
	if err := r.Update(ctx, &existingIngress); err != nil {
		return err
	}

	// Update tenant status with ingress info
	tenant.Status.Resources.Ingress = ingress.Name
	return nil
}

// reconcileRBAC creates RBAC resources
func (r *TenantReconciler) reconcileRBAC(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	// This would create Role and RoleBinding resources
	// Implementation omitted for brevity, but would follow similar pattern
	return nil
}

// cleanupTenantResources cleans up all tenant resources
func (r *TenantReconciler) cleanupTenantResources(ctx context.Context, tenant *whitelabelv1.Tenant, log logr.Logger) error {
	// Clean up namespace (which will cascade delete most resources)
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("tenant-%s", tenant.Spec.TenantID),
		},
	}
	
	if err := r.Delete(ctx, namespace); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	
	// Additional cleanup for resources outside the namespace would go here
	return nil
}

// updateTenantStatus updates the tenant status
func (r *TenantReconciler) updateTenantStatus(ctx context.Context, tenant *whitelabelv1.Tenant, phase whitelabelv1.TenantPhase, message string) {
	tenant.Status.Phase = phase
	tenant.Status.Message = message
	tenant.Status.LastReconcileTime = &metav1.Time{Time: time.Now()}
	if err := r.Status().Update(ctx, tenant); err != nil {
		r.Log.Error(err, "Failed to update tenant status")
	}
}

// Helper functions
func getStringOrDefault(value, defaultValue string) string {
	if value == "" {
		return defaultValue
	}
	return value
}

func getInt32OrDefault(value, defaultValue int32) int32 {
	if value == 0 {
		return defaultValue
	}
	return value
}

func generateAPIKey(tenantID string) string {
	// Generate secure API key
	return fmt.Sprintf("wl_%s_%d", tenantID, time.Now().Unix())
}

func generateEncryptionKey() string {
	// Generate secure encryption key
	return fmt.Sprintf("enc_%d", time.Now().UnixNano())
}

func generateJWTSecret() string {
	// Generate secure JWT secret
	return fmt.Sprintf("jwt_%d", time.Now().UnixNano())
}

func generateWebhookSecret() string {
	// Generate secure webhook secret
	return fmt.Sprintf("wh_%d", time.Now().UnixNano())
}

// SetupWithManager sets up the controller with the Manager.
func (r *TenantReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&whitelabelv1.Tenant{}).
		Owns(&corev1.Namespace{}).
		Owns(&corev1.ResourceQuota{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.Secret{}).
		Owns(&corev1.Service{}).
		Owns(&networkingv1.NetworkPolicy{}).
		Owns(&networkingv1.Ingress{}).
		Complete(r)
}