package api

import (
	"context"
	"time"
)

// BillingService defines the interface for billing operations
type BillingService interface {
	// Billing information
	GetTenantBilling(ctx context.Context, tenantID string) (*BillingInfo, error)
	UpdateTenantBilling(ctx context.Context, tenantID string, billing *BillingInfo) error
	
	// Subscription management
	CreateSubscription(ctx context.Context, tenantID, planID string) (*Subscription, error)
	UpdateSubscription(ctx context.Context, subscriptionID, planID string) (*Subscription, error)
	CancelSubscription(ctx context.Context, subscriptionID string) error
	
	// Usage tracking
	RecordUsage(ctx context.Context, tenantID string, usage *UsageRecord) error
	GetUsageHistory(ctx context.Context, tenantID string, from, to time.Time) ([]*UsageRecord, error)
	
	// Invoicing
	GenerateInvoice(ctx context.Context, tenantID string, period BillingPeriod) (*Invoice, error)
	GetInvoices(ctx context.Context, tenantID string) ([]*Invoice, error)
	ProcessPayment(ctx context.Context, invoiceID string) (*Payment, error)
}

// AnalyticsService defines the interface for analytics operations
type AnalyticsService interface {
	// System statistics
	GetSystemStats(ctx context.Context) (*SystemStats, error)
	GetSystemHealth(ctx context.Context) map[string]interface{}
	
	// Tenant analytics
	GetTenantAnalytics(ctx context.Context, tenantID string, from, to time.Time) (*TenantAnalytics, error)
	GetTenantGrowthMetrics(ctx context.Context, from, to time.Time) (*GrowthMetrics, error)
	
	// Usage analytics
	GetPlatformUsageStats(ctx context.Context, from, to time.Time) (*UsageStats, error)
	GetResourceUtilization(ctx context.Context) (*ResourceUtilization, error)
	
	// Revenue analytics
	GetRevenueMetrics(ctx context.Context, from, to time.Time) (*RevenueMetrics, error)
	GetChurnAnalysis(ctx context.Context, from, to time.Time) (*ChurnAnalysis, error)
}

// BillingInfo represents tenant billing information
type BillingInfo struct {
	TenantID        string         `json:"tenant_id"`
	PlanID          string         `json:"plan_id"`
	PlanName        string         `json:"plan_name"`
	BillingCycle    string         `json:"billing_cycle"`
	Currency        string         `json:"currency"`
	Amount          float64        `json:"amount"`
	Status          BillingStatus  `json:"status"`
	NextBillingDate time.Time      `json:"next_billing_date"`
	TrialEndsAt     *time.Time     `json:"trial_ends_at,omitempty"`
	PaymentMethod   PaymentMethod  `json:"payment_method"`
	BillingAddress  BillingAddress `json:"billing_address"`
	TaxInfo         TaxInfo        `json:"tax_info"`
	CreatedAt       time.Time      `json:"created_at"`
	UpdatedAt       time.Time      `json:"updated_at"`
}

// BillingStatus represents billing status
type BillingStatus string

const (
	BillingStatusActive    BillingStatus = "active"
	BillingStatusPastDue   BillingStatus = "past_due"
	BillingStatusCanceled  BillingStatus = "canceled"
	BillingStatusTrial     BillingStatus = "trial"
	BillingStatusSuspended BillingStatus = "suspended"
)

// PaymentMethod represents payment method information
type PaymentMethod struct {
	Type         string `json:"type"` // card, bank_account, paypal
	Last4        string `json:"last4,omitempty"`
	Brand        string `json:"brand,omitempty"`
	ExpiryMonth  int    `json:"expiry_month,omitempty"`
	ExpiryYear   int    `json:"expiry_year,omitempty"`
	BankName     string `json:"bank_name,omitempty"`
	AccountLast4 string `json:"account_last4,omitempty"`
}

// BillingAddress represents billing address
type BillingAddress struct {
	Line1      string `json:"line1"`
	Line2      string `json:"line2,omitempty"`
	City       string `json:"city"`
	State      string `json:"state"`
	PostalCode string `json:"postal_code"`
	Country    string `json:"country"`
}

// TaxInfo represents tax information
type TaxInfo struct {
	TaxID       string  `json:"tax_id,omitempty"`
	TaxRate     float64 `json:"tax_rate"`
	TaxExempt   bool    `json:"tax_exempt"`
	TaxType     string  `json:"tax_type,omitempty"` // VAT, GST, Sales Tax
}

// Subscription represents a billing subscription
type Subscription struct {
	ID              string              `json:"id"`
	TenantID        string              `json:"tenant_id"`
	PlanID          string              `json:"plan_id"`
	Status          SubscriptionStatus  `json:"status"`
	CurrentPeriodStart time.Time        `json:"current_period_start"`
	CurrentPeriodEnd   time.Time        `json:"current_period_end"`
	CancelAtPeriodEnd  bool             `json:"cancel_at_period_end"`
	CreatedAt       time.Time           `json:"created_at"`
	UpdatedAt       time.Time           `json:"updated_at"`
	Items           []SubscriptionItem  `json:"items"`
}

// SubscriptionStatus represents subscription status
type SubscriptionStatus string

const (
	SubscriptionStatusActive    SubscriptionStatus = "active"
	SubscriptionStatusPastDue   SubscriptionStatus = "past_due"
	SubscriptionStatusCanceled  SubscriptionStatus = "canceled"
	SubscriptionStatusUnpaid    SubscriptionStatus = "unpaid"
)

// SubscriptionItem represents an item in a subscription
type SubscriptionItem struct {
	ID       string  `json:"id"`
	PriceID  string  `json:"price_id"`
	Quantity int     `json:"quantity"`
	Amount   float64 `json:"amount"`
}

// UsageRecord represents a usage record for billing
type UsageRecord struct {
	TenantID    string    `json:"tenant_id"`
	MetricName  string    `json:"metric_name"`
	Quantity    int64     `json:"quantity"`
	Timestamp   time.Time `json:"timestamp"`
	Description string    `json:"description,omitempty"`
}

// BillingPeriod represents a billing period
type BillingPeriod struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// Invoice represents a billing invoice
type Invoice struct {
	ID              string        `json:"id"`
	TenantID        string        `json:"tenant_id"`
	Number          string        `json:"number"`
	Status          InvoiceStatus `json:"status"`
	Amount          float64       `json:"amount"`
	Currency        string        `json:"currency"`
	TaxAmount       float64       `json:"tax_amount"`
	TotalAmount     float64       `json:"total_amount"`
	Description     string        `json:"description"`
	PeriodStart     time.Time     `json:"period_start"`
	PeriodEnd       time.Time     `json:"period_end"`
	IssuedAt        time.Time     `json:"issued_at"`
	DueAt           time.Time     `json:"due_at"`
	PaidAt          *time.Time    `json:"paid_at,omitempty"`
	Items           []InvoiceItem `json:"items"`
	PaymentAttempts []PaymentAttempt `json:"payment_attempts"`
}

// InvoiceStatus represents invoice status
type InvoiceStatus string

const (
	InvoiceStatusDraft     InvoiceStatus = "draft"
	InvoiceStatusOpen      InvoiceStatus = "open"
	InvoiceStatusPaid      InvoiceStatus = "paid"
	InvoiceStatusVoid      InvoiceStatus = "void"
	InvoiceStatusUncollectible InvoiceStatus = "uncollectible"
)

// InvoiceItem represents an item in an invoice
type InvoiceItem struct {
	ID          string  `json:"id"`
	Description string  `json:"description"`
	Quantity    int     `json:"quantity"`
	UnitAmount  float64 `json:"unit_amount"`
	Amount      float64 `json:"amount"`
}

// PaymentAttempt represents a payment attempt
type PaymentAttempt struct {
	ID          string        `json:"id"`
	Amount      float64       `json:"amount"`
	Status      PaymentStatus `json:"status"`
	AttemptedAt time.Time     `json:"attempted_at"`
	FailureCode string        `json:"failure_code,omitempty"`
	FailureMessage string     `json:"failure_message,omitempty"`
}

// Payment represents a payment
type Payment struct {
	ID          string        `json:"id"`
	InvoiceID   string        `json:"invoice_id"`
	Amount      float64       `json:"amount"`
	Currency    string        `json:"currency"`
	Status      PaymentStatus `json:"status"`
	ProcessedAt time.Time     `json:"processed_at"`
	Method      PaymentMethod `json:"method"`
	TransactionID string      `json:"transaction_id,omitempty"`
}

// PaymentStatus represents payment status
type PaymentStatus string

const (
	PaymentStatusPending   PaymentStatus = "pending"
	PaymentStatusSucceeded PaymentStatus = "succeeded"
	PaymentStatusFailed    PaymentStatus = "failed"
	PaymentStatusCanceled  PaymentStatus = "canceled"
)

// SystemStats represents overall system statistics
type SystemStats struct {
	TotalTenants       int64   `json:"total_tenants"`
	ActiveTenants      int64   `json:"active_tenants"`
	TrialTenants       int64   `json:"trial_tenants"`
	PaidTenants        int64   `json:"paid_tenants"`
	SuspendedTenants   int64   `json:"suspended_tenants"`
	MonthlyRevenue     float64 `json:"monthly_revenue"`
	AverageRevenuePer  float64 `json:"average_revenue_per_tenant"`
	ChurnRate          float64 `json:"churn_rate"`
	GrowthRate         float64 `json:"growth_rate"`
	TotalAPIRequests   int64   `json:"total_api_requests"`
	AverageResponseTime float64 `json:"average_response_time"`
	UptimePercent      float64 `json:"uptime_percent"`
	StorageUsageGB     float64 `json:"storage_usage_gb"`
	BandwidthUsageGB   float64 `json:"bandwidth_usage_gb"`
	Timestamp          time.Time `json:"timestamp"`
}

// TenantAnalytics represents analytics data for a specific tenant
type TenantAnalytics struct {
	TenantID            string    `json:"tenant_id"`
	ActiveUsers         int64     `json:"active_users"`
	MonthlyActiveUsers  int64     `json:"monthly_active_users"`
	SessionCount        int64     `json:"session_count"`
	AverageSessionTime  time.Duration `json:"average_session_time"`
	APIRequestCount     int64     `json:"api_request_count"`
	ErrorRate           float64   `json:"error_rate"`
	FeatureUsage        map[string]int64 `json:"feature_usage"`
	StorageUsageBytes   int64     `json:"storage_usage_bytes"`
	BandwidthUsageBytes int64     `json:"bandwidth_usage_bytes"`
	Revenue             float64   `json:"revenue"`
	LastActivityAt      time.Time `json:"last_activity_at"`
	Timestamp           time.Time `json:"timestamp"`
}

// GrowthMetrics represents platform growth metrics
type GrowthMetrics struct {
	Period              string    `json:"period"`
	NewTenants          int64     `json:"new_tenants"`
	ChurnedTenants      int64     `json:"churned_tenants"`
	NetGrowth           int64     `json:"net_growth"`
	GrowthRate          float64   `json:"growth_rate"`
	ChurnRate           float64   `json:"churn_rate"`
	RevenueGrowth       float64   `json:"revenue_growth"`
	AverageTimeToValue  time.Duration `json:"average_time_to_value"`
	ConversionRate      float64   `json:"conversion_rate"`
	Timestamp           time.Time `json:"timestamp"`
}

// UsageStats represents platform usage statistics
type UsageStats struct {
	Period                string    `json:"period"`
	TotalAPIRequests      int64     `json:"total_api_requests"`
	UniqueAPIConsumers    int64     `json:"unique_api_consumers"`
	AverageResponseTime   float64   `json:"average_response_time"`
	ErrorCount            int64     `json:"error_count"`
	ErrorRate             float64   `json:"error_rate"`
	DataTransferredGB     float64   `json:"data_transferred_gb"`
	PopularEndpoints      []EndpointStats `json:"popular_endpoints"`
	GeographicDistribution map[string]int64 `json:"geographic_distribution"`
	DeviceTypes           map[string]int64 `json:"device_types"`
	Timestamp             time.Time `json:"timestamp"`
}

// EndpointStats represents statistics for API endpoints
type EndpointStats struct {
	Endpoint        string  `json:"endpoint"`
	RequestCount    int64   `json:"request_count"`
	AverageResponseTime float64 `json:"average_response_time"`
	ErrorRate       float64 `json:"error_rate"`
}

// ResourceUtilization represents resource utilization metrics
type ResourceUtilization struct {
	CPUUsagePercent     float64   `json:"cpu_usage_percent"`
	MemoryUsagePercent  float64   `json:"memory_usage_percent"`
	StorageUsagePercent float64   `json:"storage_usage_percent"`
	NetworkUsagePercent float64   `json:"network_usage_percent"`
	DatabaseConnections int       `json:"database_connections"`
	CacheHitRate        float64   `json:"cache_hit_rate"`
	QueueDepth          int       `json:"queue_depth"`
	ActiveTasks         int       `json:"active_tasks"`
	Timestamp           time.Time `json:"timestamp"`
}

// RevenueMetrics represents revenue analytics
type RevenueMetrics struct {
	Period                  string             `json:"period"`
	TotalRevenue           float64            `json:"total_revenue"`
	RecurringRevenue       float64            `json:"recurring_revenue"`
	OneTimeRevenue         float64            `json:"one_time_revenue"`
	AverageRevenuePer      float64            `json:"average_revenue_per_tenant"`
	RevenueGrowthRate      float64            `json:"revenue_growth_rate"`
	RevenueByPlan          map[string]float64 `json:"revenue_by_plan"`
	RevenueByGeography     map[string]float64 `json:"revenue_by_geography"`
	MonthlyRecurringRevenue float64           `json:"monthly_recurring_revenue"`
	AnnualRecurringRevenue  float64           `json:"annual_recurring_revenue"`
	Timestamp              time.Time          `json:"timestamp"`
}

// ChurnAnalysis represents churn analysis data
type ChurnAnalysis struct {
	Period                string             `json:"period"`
	ChurnRate             float64            `json:"churn_rate"`
	RevenueChurnRate      float64            `json:"revenue_churn_rate"`
	ChurnedTenants        int64              `json:"churned_tenants"`
	ChurnReasons          map[string]int64   `json:"churn_reasons"`
	ChurnByPlan           map[string]float64 `json:"churn_by_plan"`
	ChurnByGeography      map[string]float64 `json:"churn_by_geography"`
	AverageLifetime       time.Duration      `json:"average_lifetime"`
	PredictedChurnRisk    []ChurnRiskTenant  `json:"predicted_churn_risk"`
	RetentionRate         float64            `json:"retention_rate"`
	Timestamp             time.Time          `json:"timestamp"`
}

// ChurnRiskTenant represents a tenant at risk of churning
type ChurnRiskTenant struct {
	TenantID    string  `json:"tenant_id"`
	TenantName  string  `json:"tenant_name"`
	RiskScore   float64 `json:"risk_score"`
	RiskFactors []string `json:"risk_factors"`
	LastActive  time.Time `json:"last_active"`
}

// SupportTicket represents a support ticket
type SupportTicket struct {
	ID          string           `json:"id"`
	TenantID    string           `json:"tenant_id"`
	Subject     string           `json:"subject"`
	Description string           `json:"description"`
	Status      TicketStatus     `json:"status"`
	Priority    TicketPriority   `json:"priority"`
	Category    string           `json:"category"`
	AssignedTo  string           `json:"assigned_to,omitempty"`
	CreatedBy   string           `json:"created_by"`
	CreatedAt   time.Time        `json:"created_at"`
	UpdatedAt   time.Time        `json:"updated_at"`
	ResolvedAt  *time.Time       `json:"resolved_at,omitempty"`
	Messages    []TicketMessage  `json:"messages"`
	Tags        []string         `json:"tags"`
}

// TicketStatus represents support ticket status
type TicketStatus string

const (
	TicketStatusOpen       TicketStatus = "open"
	TicketStatusInProgress TicketStatus = "in_progress"
	TicketStatusResolved   TicketStatus = "resolved"
	TicketStatusClosed     TicketStatus = "closed"
)

// TicketPriority represents support ticket priority
type TicketPriority string

const (
	TicketPriorityLow      TicketPriority = "low"
	TicketPriorityMedium   TicketPriority = "medium"
	TicketPriorityHigh     TicketPriority = "high"
	TicketPriorityCritical TicketPriority = "critical"
)

// TicketMessage represents a message in a support ticket
type TicketMessage struct {
	ID        string    `json:"id"`
	AuthorID  string    `json:"author_id"`
	AuthorName string   `json:"author_name"`
	Content   string    `json:"content"`
	IsInternal bool     `json:"is_internal"`
	CreatedAt time.Time `json:"created_at"`
	Attachments []string `json:"attachments"`
}