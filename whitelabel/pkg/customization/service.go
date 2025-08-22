package customization

import (
	"context"
	"fmt"
	"html/template"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// Service handles customization operations
type Service struct {
	repo           Repository
	templateEngine TemplateEngine
	logger         *zap.Logger
}

// NewService creates a new customization service
func NewService(repo Repository, templateEngine TemplateEngine, logger *zap.Logger) *Service {
	return &Service{
		repo:           repo,
		templateEngine: templateEngine,
		logger:         logger,
	}
}

// ApplyTheme applies a theme configuration to a tenant
func (s *Service) ApplyTheme(ctx context.Context, tenantID string, theme ThemeConfig) error {
	s.logger.Info("Applying theme to tenant", zap.String("tenant_id", tenantID))
	
	// Validate theme configuration
	if err := s.validateThemeConfig(theme); err != nil {
		return fmt.Errorf("invalid theme configuration: %w", err)
	}
	
	// Get or create UI customization
	customization, err := s.repo.GetUICustomization(ctx, tenantID)
	if err != nil {
		// Create new customization if none exists
		customization = &UICustomization{
			ID:       uuid.New().String(),
			TenantID: tenantID,
			Theme:    theme,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		return s.repo.CreateUICustomization(ctx, customization)
	}
	
	// Update existing theme
	customization.Theme = theme
	customization.UpdatedAt = time.Now()
	
	return s.repo.UpdateUICustomization(ctx, customization)
}

// ApplyBranding applies branding configuration to a tenant
func (s *Service) ApplyBranding(ctx context.Context, tenantID string, branding BrandingConfig) error {
	s.logger.Info("Applying branding to tenant", zap.String("tenant_id", tenantID))
	
	// Validate branding configuration
	if err := s.validateBrandingConfig(branding); err != nil {
		return fmt.Errorf("invalid branding configuration: %w", err)
	}
	
	// Get or create UI customization
	customization, err := s.repo.GetUICustomization(ctx, tenantID)
	if err != nil {
		customization = &UICustomization{
			ID:       uuid.New().String(),
			TenantID: tenantID,
			Branding: branding,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		return s.repo.CreateUICustomization(ctx, customization)
	}
	
	// Update existing branding
	customization.Branding = branding
	customization.UpdatedAt = time.Now()
	
	return s.repo.UpdateUICustomization(ctx, customization)
}

// SetFeatureToggle enables/disables a feature for a tenant
func (s *Service) SetFeatureToggle(ctx context.Context, tenantID, featureID string, enabled bool, value interface{}) error {
	s.logger.Info("Setting feature toggle", 
		zap.String("tenant_id", tenantID),
		zap.String("feature_id", featureID),
		zap.Bool("enabled", enabled))
	
	// Check if feature exists
	feature, err := s.repo.GetFeatureToggle(ctx, featureID)
	if err != nil {
		return fmt.Errorf("feature not found: %w", err)
	}
	
	// Validate feature value type
	if err := s.validateFeatureValue(feature, value); err != nil {
		return fmt.Errorf("invalid feature value: %w", err)
	}
	
	// Get existing tenant feature or create new one
	tenantFeature, err := s.repo.GetTenantFeature(ctx, tenantID, featureID)
	if err != nil {
		tenantFeature = &TenantFeature{
			ID:        uuid.New().String(),
			TenantID:  tenantID,
			FeatureID: featureID,
			Value:     value,
			Enabled:   enabled,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		return s.repo.CreateTenantFeature(ctx, tenantFeature)
	}
	
	// Update existing feature
	tenantFeature.Enabled = enabled
	tenantFeature.Value = value
	tenantFeature.UpdatedAt = time.Now()
	
	return s.repo.UpdateTenantFeature(ctx, tenantFeature)
}

// GetTenantFeatures retrieves all feature settings for a tenant
func (s *Service) GetTenantFeatures(ctx context.Context, tenantID string) (map[string]*TenantFeature, error) {
	features, err := s.repo.GetTenantFeatures(ctx, tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenant features: %w", err)
	}
	
	result := make(map[string]*TenantFeature)
	for _, feature := range features {
		result[feature.FeatureID] = feature
	}
	
	return result, nil
}

// IsFeatureEnabled checks if a feature is enabled for a tenant
func (s *Service) IsFeatureEnabled(ctx context.Context, tenantID, featureID string) (bool, error) {
	tenantFeature, err := s.repo.GetTenantFeature(ctx, tenantID, featureID)
	if err != nil {
		// If tenant feature doesn't exist, check default value
		feature, err := s.repo.GetFeatureToggle(ctx, featureID)
		if err != nil {
			return false, fmt.Errorf("feature not found: %w", err)
		}
		
		// Return default value as boolean
		if defaultVal, ok := feature.DefaultValue.(bool); ok {
			return defaultVal, nil
		}
		return false, nil
	}
	
	return tenantFeature.Enabled, nil
}

// GetFeatureValue gets the value of a feature for a tenant
func (s *Service) GetFeatureValue(ctx context.Context, tenantID, featureID string) (interface{}, error) {
	tenantFeature, err := s.repo.GetTenantFeature(ctx, tenantID, featureID)
	if err != nil {
		// If tenant feature doesn't exist, return default value
		feature, err := s.repo.GetFeatureToggle(ctx, featureID)
		if err != nil {
			return nil, fmt.Errorf("feature not found: %w", err)
		}
		return feature.DefaultValue, nil
	}
	
	return tenantFeature.Value, nil
}

// CreateEmailTemplate creates a custom email template for a tenant
func (s *Service) CreateEmailTemplate(ctx context.Context, tenantID string, templateType EmailTemplateType, subject, htmlContent, textContent string, variables []string) (*EmailTemplate, error) {
	s.logger.Info("Creating email template",
		zap.String("tenant_id", tenantID),
		zap.String("type", string(templateType)))
	
	// Validate template content
	if err := s.validateEmailTemplate(htmlContent, textContent, variables); err != nil {
		return nil, fmt.Errorf("invalid email template: %w", err)
	}
	
	template := &EmailTemplate{
		ID:          uuid.New().String(),
		TenantID:    tenantID,
		Type:        templateType,
		Subject:     subject,
		HTMLContent: htmlContent,
		TextContent: textContent,
		Variables:   variables,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	
	if err := s.repo.CreateEmailTemplate(ctx, template); err != nil {
		return nil, fmt.Errorf("failed to create email template: %w", err)
	}
	
	return template, nil
}

// RenderEmailTemplate renders an email template with provided data
func (s *Service) RenderEmailTemplate(ctx context.Context, tenantID string, templateType EmailTemplateType, data map[string]interface{}) (subject, htmlContent, textContent string, err error) {
	template, err := s.repo.GetEmailTemplate(ctx, tenantID, templateType)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to get email template: %w", err)
	}
	
	// Render subject
	subject, err = s.templateEngine.RenderString(template.Subject, data)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to render subject: %w", err)
	}
	
	// Render HTML content
	htmlContent, err = s.templateEngine.RenderString(template.HTMLContent, data)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to render HTML content: %w", err)
	}
	
	// Render text content
	textContent, err = s.templateEngine.RenderString(template.TextContent, data)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to render text content: %w", err)
	}
	
	return subject, htmlContent, textContent, nil
}

// SetupOnboarding creates a customized onboarding flow for a tenant
func (s *Service) SetupOnboarding(ctx context.Context, tenantID, industry, useCase string) (*OnboardingConfig, error) {
	s.logger.Info("Setting up onboarding",
		zap.String("tenant_id", tenantID),
		zap.String("industry", industry),
		zap.String("use_case", useCase))
	
	// Get industry defaults if available
	var steps []OnboardingStep
	var customizations OnboardingCustomizations
	
	if industry != "" {
		defaults, err := s.repo.GetIndustryDefaults(ctx, industry)
		if err == nil {
			steps = defaults.OnboardingSteps
			customizations = OnboardingCustomizations{
				ShowIndustrySpecificTips: true,
				RecommendedFeatures:      []string{}, // populated from defaults
				SampleData:              true,
				DefaultSettings:          defaults.FeatureDefaults,
			}
		}
	}
	
	// Use default steps if no industry-specific ones found
	if len(steps) == 0 {
		steps = s.getDefaultOnboardingSteps(useCase)
	}
	
	config := &OnboardingConfig{
		ID:             uuid.New().String(),
		TenantID:       tenantID,
		Steps:          steps,
		WelcomeMessage: s.getWelcomeMessage(industry, useCase),
		CompletionURL:  "/dashboard",
		SkipEnabled:    true,
		Industry:       industry,
		UseCase:        useCase,
		Customizations: customizations,
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}
	
	if err := s.repo.CreateOnboardingConfig(ctx, config); err != nil {
		return nil, fmt.Errorf("failed to create onboarding config: %w", err)
	}
	
	return config, nil
}

// GetTenantCustomization retrieves all customizations for a tenant
func (s *Service) GetTenantCustomization(ctx context.Context, tenantID string) (*UICustomization, error) {
	return s.repo.GetUICustomization(ctx, tenantID)
}

// GenerateCSS generates CSS for tenant's theme configuration
func (s *Service) GenerateCSS(ctx context.Context, tenantID string) (string, error) {
	customization, err := s.repo.GetUICustomization(ctx, tenantID)
	if err != nil {
		return "", fmt.Errorf("failed to get UI customization: %w", err)
	}
	
	var css strings.Builder
	theme := customization.Theme
	
	// Generate CSS variables
	css.WriteString(":root {\n")
	css.WriteString(fmt.Sprintf("  --primary-color: %s;\n", theme.PrimaryColor))
	css.WriteString(fmt.Sprintf("  --secondary-color: %s;\n", theme.SecondaryColor))
	css.WriteString(fmt.Sprintf("  --accent-color: %s;\n", theme.AccentColor))
	css.WriteString(fmt.Sprintf("  --background-color: %s;\n", theme.BackgroundColor))
	css.WriteString(fmt.Sprintf("  --text-color: %s;\n", theme.TextColor))
	css.WriteString(fmt.Sprintf("  --link-color: %s;\n", theme.LinkColor))
	css.WriteString(fmt.Sprintf("  --font-family: %s;\n", theme.FontFamily))
	css.WriteString(fmt.Sprintf("  --font-size: %s;\n", theme.FontSize))
	css.WriteString(fmt.Sprintf("  --border-radius: %s;\n", theme.BorderRadius))
	css.WriteString(fmt.Sprintf("  --box-shadow: %s;\n", theme.BoxShadow))
	
	// Add custom properties
	for key, value := range theme.CustomProperties {
		css.WriteString(fmt.Sprintf("  --%s: %s;\n", key, value))
	}
	
	css.WriteString("}\n\n")
	
	// Generate component styles
	css.WriteString("body {\n")
	css.WriteString("  font-family: var(--font-family);\n")
	css.WriteString("  font-size: var(--font-size);\n")
	css.WriteString("  color: var(--text-color);\n")
	css.WriteString("  background-color: var(--background-color);\n")
	css.WriteString("}\n\n")
	
	css.WriteString("a {\n")
	css.WriteString("  color: var(--link-color);\n")
	css.WriteString("}\n\n")
	
	css.WriteString(".btn-primary {\n")
	css.WriteString("  background-color: var(--primary-color);\n")
	css.WriteString("  border-color: var(--primary-color);\n")
	css.WriteString("  border-radius: var(--border-radius);\n")
	css.WriteString("}\n\n")
	
	css.WriteString(".btn-secondary {\n")
	css.WriteString("  background-color: var(--secondary-color);\n")
	css.WriteString("  border-color: var(--secondary-color);\n")
	css.WriteString("  border-radius: var(--border-radius);\n")
	css.WriteString("}\n\n")
	
	css.WriteString(".card {\n")
	css.WriteString("  box-shadow: var(--box-shadow);\n")
	css.WriteString("  border-radius: var(--border-radius);\n")
	css.WriteString("}\n\n")
	
	// Dark mode styles
	if theme.DarkMode {
		css.WriteString("@media (prefers-color-scheme: dark) {\n")
		css.WriteString("  :root {\n")
		css.WriteString("    --background-color: #1a1a1a;\n")
		css.WriteString("    --text-color: #ffffff;\n")
		css.WriteString("  }\n")
		css.WriteString("}\n\n")
	}
	
	// Add custom CSS
	if theme.CustomCSS != "" {
		css.WriteString("/* Custom CSS */\n")
		css.WriteString(theme.CustomCSS)
		css.WriteString("\n")
	}
	
	return css.String(), nil
}

// Validation helper methods

func (s *Service) validateThemeConfig(theme ThemeConfig) error {
	if theme.PrimaryColor == "" {
		return fmt.Errorf("primary color is required")
	}
	if theme.FontFamily == "" {
		return fmt.Errorf("font family is required")
	}
	return nil
}

func (s *Service) validateBrandingConfig(branding BrandingConfig) error {
	if branding.CompanyName == "" {
		return fmt.Errorf("company name is required")
	}
	return nil
}

func (s *Service) validateFeatureValue(feature *FeatureToggle, value interface{}) error {
	switch feature.Type {
	case FeatureTypeBoolean:
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected boolean value")
		}
	case FeatureTypeString:
		if _, ok := value.(string); !ok {
			return fmt.Errorf("expected string value")
		}
	case FeatureTypeNumber:
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("expected number value")
		}
	case FeatureTypeJSON:
		// JSON values are typically stored as maps or slices
		// Additional validation can be added here
	}
	return nil
}

func (s *Service) validateEmailTemplate(htmlContent, textContent string, variables []string) error {
	// Validate HTML template
	if htmlContent != "" {
		_, err := template.New("html").Parse(htmlContent)
		if err != nil {
			return fmt.Errorf("invalid HTML template: %w", err)
		}
	}
	
	// Validate text template
	if textContent != "" {
		_, err := template.New("text").Parse(textContent)
		if err != nil {
			return fmt.Errorf("invalid text template: %w", err)
		}
	}
	
	return nil
}

func (s *Service) getDefaultOnboardingSteps(useCase string) []OnboardingStep {
	steps := []OnboardingStep{
		{
			ID:          "welcome",
			Title:       "Welcome",
			Description: "Welcome to your personalized platform",
			Type:        StepTypeWelcome,
			Required:    true,
			Order:       1,
		},
		{
			ID:          "profile_setup",
			Title:       "Setup Your Profile",
			Description: "Complete your profile information",
			Type:        StepTypeProfileSetup,
			Required:    true,
			Order:       2,
		},
		{
			ID:          "team_invite",
			Title:       "Invite Your Team",
			Description: "Invite team members to collaborate",
			Type:        StepTypeTeamInvite,
			Required:    false,
			Order:       3,
		},
		{
			ID:          "tutorial",
			Title:       "Quick Tutorial",
			Description: "Learn the basics of the platform",
			Type:        StepTypeTutorial,
			Required:    false,
			Order:       4,
		},
		{
			ID:          "complete",
			Title:       "Setup Complete",
			Description: "Your platform is ready to use",
			Type:        StepTypeComplete,
			Required:    true,
			Order:       5,
		},
	}
	
	// Customize based on use case
	if useCase == "enterprise" {
		steps = append(steps[:3], OnboardingStep{
			ID:          "integration",
			Title:       "Setup Integrations",
			Description: "Connect with your existing tools",
			Type:        StepTypeIntegration,
			Required:    false,
			Order:       3,
		})
	}
	
	return steps
}

func (s *Service) getWelcomeMessage(industry, useCase string) string {
	if industry != "" {
		return fmt.Sprintf("Welcome to your customized %s solution! We've tailored this experience specifically for your industry.", industry)
	}
	if useCase != "" {
		return fmt.Sprintf("Welcome! We've set up your platform for %s use cases.", useCase)
	}
	return "Welcome to your personalized white-label platform!"
}