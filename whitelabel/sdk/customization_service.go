package whitelabelsdk

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// CustomizationService provides methods for tenant customization operations
type CustomizationService struct {
	client *Client
}

// NewCustomizationService creates a new customization service
func NewCustomizationService(client *Client) *CustomizationService {
	return &CustomizationService{client: client}
}

// ThemeConfig defines the visual theme for a tenant
type ThemeConfig struct {
	PrimaryColor      string            `json:"primary_color"`
	SecondaryColor    string            `json:"secondary_color"`
	AccentColor       string            `json:"accent_color"`
	BackgroundColor   string            `json:"background_color"`
	TextColor         string            `json:"text_color"`
	LinkColor         string            `json:"link_color"`
	FontFamily        string            `json:"font_family"`
	FontSize          string            `json:"font_size"`
	BorderRadius      string            `json:"border_radius"`
	BoxShadow         string            `json:"box_shadow"`
	LogoURL           string            `json:"logo_url"`
	FaviconURL        string            `json:"favicon_url"`
	CustomCSS         string            `json:"custom_css"`
	DarkMode          bool              `json:"dark_mode"`
	CustomProperties  map[string]string `json:"custom_properties"`
}

// BrandingConfig defines branding elements for a tenant
type BrandingConfig struct {
	CompanyName        string            `json:"company_name"`
	CompanyDescription string            `json:"company_description"`
	LogoURL            string            `json:"logo_url"`
	LogoWidth          int               `json:"logo_width"`
	LogoHeight         int               `json:"logo_height"`
	FaviconURL         string            `json:"favicon_url"`
	HeaderText         string            `json:"header_text"`
	FooterText         string            `json:"footer_text"`
	ContactEmail       string            `json:"contact_email"`
	SupportURL         string            `json:"support_url"`
	TermsURL           string            `json:"terms_url"`
	PrivacyURL         string            `json:"privacy_url"`
	SocialMedia        SocialMediaConfig `json:"social_media"`
	CustomMetaTags     map[string]string `json:"custom_meta_tags"`
}

// SocialMediaConfig defines social media links
type SocialMediaConfig struct {
	Facebook  string `json:"facebook,omitempty"`
	Twitter   string `json:"twitter,omitempty"`
	LinkedIn  string `json:"linkedin,omitempty"`
	Instagram string `json:"instagram,omitempty"`
	YouTube   string `json:"youtube,omitempty"`
	GitHub    string `json:"github,omitempty"`
}

// UICustomization defines UI customizations for a tenant
type UICustomization struct {
	ID               string            `json:"id"`
	TenantID         string            `json:"tenant_id"`
	Theme            ThemeConfig       `json:"theme"`
	Branding         BrandingConfig    `json:"branding"`
	Layout           LayoutConfig      `json:"layout"`
	Navigation       NavigationConfig  `json:"navigation"`
	CustomComponents []CustomComponent `json:"custom_components"`
	CreatedAt        time.Time         `json:"created_at"`
	UpdatedAt        time.Time         `json:"updated_at"`
}

// LayoutConfig defines layout customizations
type LayoutConfig struct {
	Sidebar      SidebarConfig `json:"sidebar"`
	Header       HeaderConfig  `json:"header"`
	Footer       FooterConfig  `json:"footer"`
	ContentWidth string        `json:"content_width"`
	Spacing      string        `json:"spacing"`
}

// SidebarConfig defines sidebar customizations
type SidebarConfig struct {
	Position         string `json:"position"` // left, right, hidden
	Width            string `json:"width"`
	Collapsible      bool   `json:"collapsible"`
	DefaultCollapsed bool   `json:"default_collapsed"`
}

// HeaderConfig defines header customizations
type HeaderConfig struct {
	Height     string `json:"height"`
	ShowLogo   bool   `json:"show_logo"`
	ShowTitle  bool   `json:"show_title"`
	ShowSearch bool   `json:"show_search"`
	ShowUser   bool   `json:"show_user"`
}

// FooterConfig defines footer customizations
type FooterConfig struct {
	Show    bool         `json:"show"`
	Height  string       `json:"height"`
	Content string       `json:"content"`
	Links   []FooterLink `json:"links"`
}

// FooterLink defines a footer link
type FooterLink struct {
	Text string `json:"text"`
	URL  string `json:"url"`
}

// NavigationConfig defines navigation customizations
type NavigationConfig struct {
	MenuItems       []MenuItem `json:"menu_items"`
	ShowIcons       bool       `json:"show_icons"`
	ShowBadges      bool       `json:"show_badges"`
	GroupByCategory bool       `json:"group_by_category"`
}

// MenuItem defines a navigation menu item
type MenuItem struct {
	ID              string     `json:"id"`
	Label           string     `json:"label"`
	Icon            string     `json:"icon"`
	URL             string     `json:"url"`
	Badge           string     `json:"badge"`
	Category        string     `json:"category"`
	Order           int        `json:"order"`
	Children        []MenuItem `json:"children"`
	Visible         bool       `json:"visible"`
	RequiredFeature string     `json:"required_feature"`
}

// CustomComponent defines a custom UI component
type CustomComponent struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Position    string                 `json:"position"`
	HTMLContent string                 `json:"html_content"`
	CSSContent  string                 `json:"css_content"`
	JSContent   string                 `json:"js_content"`
	Config      map[string]interface{} `json:"config"`
	Enabled     bool                   `json:"enabled"`
}

// EmailTemplate defines customizable email templates
type EmailTemplate struct {
	ID          string            `json:"id"`
	TenantID    string            `json:"tenant_id"`
	Type        EmailTemplateType `json:"type"`
	Subject     string            `json:"subject"`
	HTMLContent string            `json:"html_content"`
	TextContent string            `json:"text_content"`
	Variables   []string          `json:"variables"`
	IsDefault   bool              `json:"is_default"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// EmailTemplateType defines the type of email template
type EmailTemplateType string

const (
	EmailTypeWelcome       EmailTemplateType = "welcome"
	EmailTypePasswordReset EmailTemplateType = "password_reset"
	EmailTypeInvitation    EmailTemplateType = "invitation"
	EmailTypeNotification  EmailTemplateType = "notification"
	EmailTypeBilling       EmailTemplateType = "billing"
	EmailTypeSupport       EmailTemplateType = "support"
)

// OnboardingConfig defines the onboarding flow for tenants
type OnboardingConfig struct {
	ID             string                   `json:"id"`
	TenantID       string                   `json:"tenant_id"`
	Steps          []OnboardingStep         `json:"steps"`
	WelcomeMessage string                   `json:"welcome_message"`
	CompletionURL  string                   `json:"completion_url"`
	SkipEnabled    bool                     `json:"skip_enabled"`
	Industry       string                   `json:"industry"`
	UseCase        string                   `json:"use_case"`
	Customizations OnboardingCustomizations `json:"customizations"`
	CreatedAt      time.Time                `json:"created_at"`
	UpdatedAt      time.Time                `json:"updated_at"`
}

// OnboardingStep defines a step in the onboarding process
type OnboardingStep struct {
	ID          string                 `json:"id"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Type        OnboardingStepType     `json:"type"`
	Required    bool                   `json:"required"`
	Order       int                    `json:"order"`
	Config      map[string]interface{} `json:"config"`
	Completed   bool                   `json:"completed"`
}

// OnboardingStepType defines the type of onboarding step
type OnboardingStepType string

const (
	StepTypeWelcome      OnboardingStepType = "welcome"
	StepTypeProfileSetup OnboardingStepType = "profile_setup"
	StepTypeTeamInvite   OnboardingStepType = "team_invite"
	StepTypeTutorial     OnboardingStepType = "tutorial"
	StepTypeIntegration  OnboardingStepType = "integration"
	StepTypeComplete     OnboardingStepType = "complete"
)

// OnboardingCustomizations defines industry-specific customizations
type OnboardingCustomizations struct {
	ShowIndustrySpecificTips bool                   `json:"show_industry_specific_tips"`
	RecommendedFeatures      []string               `json:"recommended_features"`
	SampleData               bool                   `json:"sample_data"`
	CustomTutorials          []string               `json:"custom_tutorials"`
	DefaultSettings          map[string]interface{} `json:"default_settings"`
}

// GetCustomization retrieves customization settings for a tenant
func (s *CustomizationService) GetCustomization(ctx context.Context, tenantID string) (*UICustomization, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var customization UICustomization
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/customization", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &customization)
	if err != nil {
		return nil, fmt.Errorf("failed to get customization: %w", err)
	}
	
	return &customization, nil
}

// ApplyTheme applies a theme configuration to a tenant
func (s *CustomizationService) ApplyTheme(ctx context.Context, tenantID string, theme *ThemeConfig) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if theme == nil {
		return fmt.Errorf("theme cannot be nil")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/theme", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, theme, nil)
	if err != nil {
		return fmt.Errorf("failed to apply theme: %w", err)
	}
	
	return nil
}

// ApplyBranding applies branding configuration to a tenant
func (s *CustomizationService) ApplyBranding(ctx context.Context, tenantID string, branding *BrandingConfig) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if branding == nil {
		return fmt.Errorf("branding cannot be nil")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/branding", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, branding, nil)
	if err != nil {
		return fmt.Errorf("failed to apply branding: %w", err)
	}
	
	return nil
}

// UpdateLayout updates layout configuration for a tenant
func (s *CustomizationService) UpdateLayout(ctx context.Context, tenantID string, layout *LayoutConfig) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if layout == nil {
		return fmt.Errorf("layout cannot be nil")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/layout", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, layout, nil)
	if err != nil {
		return fmt.Errorf("failed to update layout: %w", err)
	}
	
	return nil
}

// UpdateNavigation updates navigation configuration for a tenant
func (s *CustomizationService) UpdateNavigation(ctx context.Context, tenantID string, navigation *NavigationConfig) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if navigation == nil {
		return fmt.Errorf("navigation cannot be nil")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/navigation", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, navigation, nil)
	if err != nil {
		return fmt.Errorf("failed to update navigation: %w", err)
	}
	
	return nil
}

// CreateEmailTemplate creates a custom email template for a tenant
func (s *CustomizationService) CreateEmailTemplate(ctx context.Context, template *EmailTemplate) (*EmailTemplate, error) {
	if template == nil {
		return nil, fmt.Errorf("template cannot be nil")
	}
	if template.TenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var result EmailTemplate
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/email-templates", template.TenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, template, &result)
	if err != nil {
		return nil, fmt.Errorf("failed to create email template: %w", err)
	}
	
	return &result, nil
}

// GetEmailTemplate retrieves an email template
func (s *CustomizationService) GetEmailTemplate(ctx context.Context, tenantID string, templateType EmailTemplateType) (*EmailTemplate, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	if templateType == "" {
		return nil, fmt.Errorf("template type cannot be empty")
	}
	
	var template EmailTemplate
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/email-templates/%s", tenantID, templateType)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &template)
	if err != nil {
		return nil, fmt.Errorf("failed to get email template: %w", err)
	}
	
	return &template, nil
}

// UpdateEmailTemplate updates an email template
func (s *CustomizationService) UpdateEmailTemplate(ctx context.Context, template *EmailTemplate) (*EmailTemplate, error) {
	if template == nil {
		return nil, fmt.Errorf("template cannot be nil")
	}
	if template.TenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	if template.ID == "" {
		return nil, fmt.Errorf("template ID cannot be empty")
	}
	
	var result EmailTemplate
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/email-templates/%s", template.TenantID, template.ID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, template, &result)
	if err != nil {
		return nil, fmt.Errorf("failed to update email template: %w", err)
	}
	
	return &result, nil
}

// ListEmailTemplates lists all email templates for a tenant
func (s *CustomizationService) ListEmailTemplates(ctx context.Context, tenantID string) ([]*EmailTemplate, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var response struct {
		Templates []*EmailTemplate `json:"templates"`
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/email-templates", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &response)
	if err != nil {
		return nil, fmt.Errorf("failed to list email templates: %w", err)
	}
	
	return response.Templates, nil
}

// CreateOnboardingConfig creates onboarding configuration for a tenant
func (s *CustomizationService) CreateOnboardingConfig(ctx context.Context, config *OnboardingConfig) (*OnboardingConfig, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	if config.TenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var result OnboardingConfig
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/onboarding", config.TenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, config, &result)
	if err != nil {
		return nil, fmt.Errorf("failed to create onboarding config: %w", err)
	}
	
	return &result, nil
}

// GetOnboardingConfig retrieves onboarding configuration for a tenant
func (s *CustomizationService) GetOnboardingConfig(ctx context.Context, tenantID string) (*OnboardingConfig, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var config OnboardingConfig
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/onboarding", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to get onboarding config: %w", err)
	}
	
	return &config, nil
}

// UpdateOnboardingStep updates the completion status of an onboarding step
func (s *CustomizationService) UpdateOnboardingStep(ctx context.Context, tenantID, stepID string, completed bool) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if stepID == "" {
		return fmt.Errorf("step ID cannot be empty")
	}
	
	request := map[string]bool{"completed": completed}
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/onboarding/steps/%s", tenantID, stepID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, request, nil)
	if err != nil {
		return fmt.Errorf("failed to update onboarding step: %w", err)
	}
	
	return nil
}

// GenerateCSS generates CSS for tenant's theme configuration
func (s *CustomizationService) GenerateCSS(ctx context.Context, tenantID string) (string, error) {
	if tenantID == "" {
		return "", fmt.Errorf("tenant ID cannot be empty")
	}
	
	var response struct {
		CSS string `json:"css"`
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/css", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &response)
	if err != nil {
		return "", fmt.Errorf("failed to generate CSS: %w", err)
	}
	
	return response.CSS, nil
}

// AddCustomComponent adds a custom component to a tenant
func (s *CustomizationService) AddCustomComponent(ctx context.Context, tenantID string, component *CustomComponent) (*CustomComponent, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	if component == nil {
		return nil, fmt.Errorf("component cannot be nil")
	}
	
	var result CustomComponent
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/components", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, component, &result)
	if err != nil {
		return nil, fmt.Errorf("failed to add custom component: %w", err)
	}
	
	return &result, nil
}

// UpdateCustomComponent updates a custom component
func (s *CustomizationService) UpdateCustomComponent(ctx context.Context, tenantID, componentID string, component *CustomComponent) (*CustomComponent, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	if componentID == "" {
		return nil, fmt.Errorf("component ID cannot be empty")
	}
	if component == nil {
		return nil, fmt.Errorf("component cannot be nil")
	}
	
	var result CustomComponent
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/components/%s", tenantID, componentID)
	err := s.client.makeRequest(ctx, http.MethodPut, endpoint, component, &result)
	if err != nil {
		return nil, fmt.Errorf("failed to update custom component: %w", err)
	}
	
	return &result, nil
}

// DeleteCustomComponent deletes a custom component
func (s *CustomizationService) DeleteCustomComponent(ctx context.Context, tenantID, componentID string) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if componentID == "" {
		return fmt.Errorf("component ID cannot be empty")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/components/%s", tenantID, componentID)
	err := s.client.makeRequest(ctx, http.MethodDelete, endpoint, nil, nil)
	if err != nil {
		return fmt.Errorf("failed to delete custom component: %w", err)
	}
	
	return nil
}

// ExportCustomization exports all customization settings for a tenant
func (s *CustomizationService) ExportCustomization(ctx context.Context, tenantID string) (*UICustomization, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	
	var customization UICustomization
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/customization/export", tenantID)
	err := s.client.makeRequest(ctx, http.MethodGet, endpoint, nil, &customization)
	if err != nil {
		return nil, fmt.Errorf("failed to export customization: %w", err)
	}
	
	return &customization, nil
}

// ImportCustomization imports customization settings for a tenant
func (s *CustomizationService) ImportCustomization(ctx context.Context, tenantID string, customization *UICustomization) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	if customization == nil {
		return fmt.Errorf("customization cannot be nil")
	}
	
	endpoint := fmt.Sprintf("/api/v1/tenants/%s/customization/import", tenantID)
	err := s.client.makeRequest(ctx, http.MethodPost, endpoint, customization, nil)
	if err != nil {
		return fmt.Errorf("failed to import customization: %w", err)
	}
	
	return nil
}