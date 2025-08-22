package customization

import (
	"context"
)

// Repository defines the interface for customization data operations
type Repository interface {
	// UI Customization operations
	CreateUICustomization(ctx context.Context, customization *UICustomization) error
	GetUICustomization(ctx context.Context, tenantID string) (*UICustomization, error)
	UpdateUICustomization(ctx context.Context, customization *UICustomization) error
	DeleteUICustomization(ctx context.Context, tenantID string) error
	
	// Feature toggle operations
	CreateFeatureToggle(ctx context.Context, feature *FeatureToggle) error
	GetFeatureToggle(ctx context.Context, id string) (*FeatureToggle, error)
	GetAllFeatureToggles(ctx context.Context) ([]*FeatureToggle, error)
	UpdateFeatureToggle(ctx context.Context, feature *FeatureToggle) error
	DeleteFeatureToggle(ctx context.Context, id string) error
	
	// Tenant feature operations
	CreateTenantFeature(ctx context.Context, tenantFeature *TenantFeature) error
	GetTenantFeature(ctx context.Context, tenantID, featureID string) (*TenantFeature, error)
	GetTenantFeatures(ctx context.Context, tenantID string) ([]*TenantFeature, error)
	UpdateTenantFeature(ctx context.Context, tenantFeature *TenantFeature) error
	DeleteTenantFeature(ctx context.Context, tenantID, featureID string) error
	BulkUpdateTenantFeatures(ctx context.Context, tenantID string, features map[string]interface{}) error
	
	// Email template operations
	CreateEmailTemplate(ctx context.Context, template *EmailTemplate) error
	GetEmailTemplate(ctx context.Context, tenantID string, templateType EmailTemplateType) (*EmailTemplate, error)
	GetEmailTemplates(ctx context.Context, tenantID string) ([]*EmailTemplate, error)
	UpdateEmailTemplate(ctx context.Context, template *EmailTemplate) error
	DeleteEmailTemplate(ctx context.Context, id string) error
	
	// Onboarding configuration operations
	CreateOnboardingConfig(ctx context.Context, config *OnboardingConfig) error
	GetOnboardingConfig(ctx context.Context, tenantID string) (*OnboardingConfig, error)
	UpdateOnboardingConfig(ctx context.Context, config *OnboardingConfig) error
	DeleteOnboardingConfig(ctx context.Context, tenantID string) error
	UpdateOnboardingStep(ctx context.Context, tenantID, stepID string, completed bool) error
	
	// Industry defaults operations
	CreateIndustryDefaults(ctx context.Context, defaults *IndustryDefaults) error
	GetIndustryDefaults(ctx context.Context, industry string) (*IndustryDefaults, error)
	GetAllIndustryDefaults(ctx context.Context) ([]*IndustryDefaults, error)
	UpdateIndustryDefaults(ctx context.Context, defaults *IndustryDefaults) error
	DeleteIndustryDefaults(ctx context.Context, industry string) error
}

// TemplateEngine defines the interface for template rendering
type TemplateEngine interface {
	RenderString(templateStr string, data map[string]interface{}) (string, error)
	RenderFile(templatePath string, data map[string]interface{}) (string, error)
	RegisterFunction(name string, fn interface{}) error
}