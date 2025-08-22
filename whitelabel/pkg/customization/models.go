package customization

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
	"time"
)

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

// Value implements driver.Valuer for database storage
func (tc ThemeConfig) Value() (driver.Value, error) {
	return json.Marshal(tc)
}

// Scan implements sql.Scanner for database retrieval
func (tc *ThemeConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, tc)
}

// BrandingConfig defines branding elements for a tenant
type BrandingConfig struct {
	CompanyName       string            `json:"company_name"`
	CompanyDescription string           `json:"company_description"`
	LogoURL           string            `json:"logo_url"`
	LogoWidth         int               `json:"logo_width"`
	LogoHeight        int               `json:"logo_height"`
	FaviconURL        string            `json:"favicon_url"`
	HeaderText        string            `json:"header_text"`
	FooterText        string            `json:"footer_text"`
	ContactEmail      string            `json:"contact_email"`
	SupportURL        string            `json:"support_url"`
	TermsURL          string            `json:"terms_url"`
	PrivacyURL        string            `json:"privacy_url"`
	SocialMedia       SocialMediaConfig `json:"social_media"`
	CustomMetaTags    map[string]string `json:"custom_meta_tags"`
}

// Value implements driver.Valuer for database storage
func (bc BrandingConfig) Value() (driver.Value, error) {
	return json.Marshal(bc)
}

// Scan implements sql.Scanner for database retrieval
func (bc *BrandingConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, bc)
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

// FeatureToggle defines a feature that can be enabled/disabled per tenant
type FeatureToggle struct {
	ID          string                 `json:"id" db:"id"`
	Name        string                 `json:"name" db:"name"`
	Description string                 `json:"description" db:"description"`
	Category    string                 `json:"category" db:"category"`
	Type        FeatureType            `json:"type" db:"type"`
	DefaultValue interface{}           `json:"default_value" db:"default_value"`
	Config      map[string]interface{} `json:"config" db:"config"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" db:"updated_at"`
}

// FeatureType defines the type of feature toggle
type FeatureType string

const (
	FeatureTypeBoolean FeatureType = "boolean"
	FeatureTypeString  FeatureType = "string"
	FeatureTypeNumber  FeatureType = "number"
	FeatureTypeJSON    FeatureType = "json"
)

// TenantFeature represents a feature setting for a specific tenant
type TenantFeature struct {
	ID        string      `json:"id" db:"id"`
	TenantID  string      `json:"tenant_id" db:"tenant_id"`
	FeatureID string      `json:"feature_id" db:"feature_id"`
	Value     interface{} `json:"value" db:"value"`
	Enabled   bool        `json:"enabled" db:"enabled"`
	CreatedAt time.Time   `json:"created_at" db:"created_at"`
	UpdatedAt time.Time   `json:"updated_at" db:"updated_at"`
}

// EmailTemplate defines customizable email templates
type EmailTemplate struct {
	ID          string            `json:"id" db:"id"`
	TenantID    string            `json:"tenant_id" db:"tenant_id"`
	Type        EmailTemplateType `json:"type" db:"type"`
	Subject     string            `json:"subject" db:"subject"`
	HTMLContent string            `json:"html_content" db:"html_content"`
	TextContent string            `json:"text_content" db:"text_content"`
	Variables   []string          `json:"variables" db:"variables"`
	IsDefault   bool              `json:"is_default" db:"is_default"`
	CreatedAt   time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at" db:"updated_at"`
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
	ID             string                `json:"id" db:"id"`
	TenantID       string                `json:"tenant_id" db:"tenant_id"`
	Steps          []OnboardingStep      `json:"steps" db:"steps"`
	WelcomeMessage string                `json:"welcome_message" db:"welcome_message"`
	CompletionURL  string                `json:"completion_url" db:"completion_url"`
	SkipEnabled    bool                  `json:"skip_enabled" db:"skip_enabled"`
	Industry       string                `json:"industry" db:"industry"`
	UseCase        string                `json:"use_case" db:"use_case"`
	Customizations OnboardingCustomizations `json:"customizations" db:"customizations"`
	CreatedAt      time.Time             `json:"created_at" db:"created_at"`
	UpdatedAt      time.Time             `json:"updated_at" db:"updated_at"`
}

// OnboardingStep defines a step in the onboarding process
type OnboardingStep struct {
	ID           string                 `json:"id"`
	Title        string                 `json:"title"`
	Description  string                 `json:"description"`
	Type         OnboardingStepType     `json:"type"`
	Required     bool                   `json:"required"`
	Order        int                    `json:"order"`
	Config       map[string]interface{} `json:"config"`
	Completed    bool                   `json:"completed"`
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
	ShowIndustrySpecificTips bool     `json:"show_industry_specific_tips"`
	RecommendedFeatures      []string `json:"recommended_features"`
	SampleData               bool     `json:"sample_data"`
	CustomTutorials          []string `json:"custom_tutorials"`
	DefaultSettings          map[string]interface{} `json:"default_settings"`
}

// Value implements driver.Valuer for database storage
func (os []OnboardingStep) Value() (driver.Value, error) {
	return json.Marshal(os)
}

// Scan implements sql.Scanner for database retrieval
func (os *[]OnboardingStep) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, os)
}

// Value implements driver.Valuer for database storage
func (oc OnboardingCustomizations) Value() (driver.Value, error) {
	return json.Marshal(oc)
}

// Scan implements sql.Scanner for database retrieval
func (oc *OnboardingCustomizations) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	
	return json.Unmarshal(bytes, oc)
}

// UICustomization defines UI customizations for a tenant
type UICustomization struct {
	ID               string            `json:"id" db:"id"`
	TenantID         string            `json:"tenant_id" db:"tenant_id"`
	Theme            ThemeConfig       `json:"theme" db:"theme"`
	Branding         BrandingConfig    `json:"branding" db:"branding"`
	Layout           LayoutConfig      `json:"layout" db:"layout"`
	Navigation       NavigationConfig  `json:"navigation" db:"navigation"`
	CustomComponents []CustomComponent `json:"custom_components" db:"custom_components"`
	CreatedAt        time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt        time.Time         `json:"updated_at" db:"updated_at"`
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
	Position   string `json:"position"` // left, right, hidden
	Width      string `json:"width"`
	Collapsible bool  `json:"collapsible"`
	DefaultCollapsed bool `json:"default_collapsed"`
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
	Show      bool   `json:"show"`
	Height    string `json:"height"`
	Content   string `json:"content"`
	Links     []FooterLink `json:"links"`
}

// FooterLink defines a footer link
type FooterLink struct {
	Text string `json:"text"`
	URL  string `json:"url"`
}

// NavigationConfig defines navigation customizations
type NavigationConfig struct {
	MenuItems    []MenuItem `json:"menu_items"`
	ShowIcons    bool       `json:"show_icons"`
	ShowBadges   bool       `json:"show_badges"`
	GroupByCategory bool    `json:"group_by_category"`
}

// MenuItem defines a navigation menu item
type MenuItem struct {
	ID       string     `json:"id"`
	Label    string     `json:"label"`
	Icon     string     `json:"icon"`
	URL      string     `json:"url"`
	Badge    string     `json:"badge"`
	Category string     `json:"category"`
	Order    int        `json:"order"`
	Children []MenuItem `json:"children"`
	Visible  bool       `json:"visible"`
	RequiredFeature string `json:"required_feature"`
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

// Value implementations for complex types
func (lc LayoutConfig) Value() (driver.Value, error) {
	return json.Marshal(lc)
}

func (lc *LayoutConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	return json.Unmarshal(bytes, lc)
}

func (nc NavigationConfig) Value() (driver.Value, error) {
	return json.Marshal(nc)
}

func (nc *NavigationConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	return json.Unmarshal(bytes, nc)
}

func (cc []CustomComponent) Value() (driver.Value, error) {
	return json.Marshal(cc)
}

func (cc *[]CustomComponent) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	return json.Unmarshal(bytes, cc)
}

// IndustryDefaults defines default settings for specific industries
type IndustryDefaults struct {
	ID              string            `json:"id" db:"id"`
	Industry        string            `json:"industry" db:"industry"`
	DisplayName     string            `json:"display_name" db:"display_name"`
	Description     string            `json:"description" db:"description"`
	ThemeConfig     ThemeConfig       `json:"theme_config" db:"theme_config"`
	FeatureDefaults map[string]interface{} `json:"feature_defaults" db:"feature_defaults"`
	OnboardingSteps []OnboardingStep  `json:"onboarding_steps" db:"onboarding_steps"`
	RecommendedIntegrations []string `json:"recommended_integrations" db:"recommended_integrations"`
	CreatedAt       time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt       time.Time         `json:"updated_at" db:"updated_at"`
}

// Value implementations for IndustryDefaults
func (fd map[string]interface{}) Value() (driver.Value, error) {
	return json.Marshal(fd)
}

func (fd *map[string]interface{}) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("type assertion to []byte failed")
	}
	return json.Unmarshal(bytes, fd)
}