Pod::Spec.new do |spec|
  spec.name             = 'ArbitrationSDK'
  spec.version          = '1.0.0'
  spec.summary          = 'iOS SDK for arbitration clause detection and analysis'
  spec.description      = <<-DESC
                          ArbitrationSDK provides comprehensive tools for detecting, analyzing, and visualizing
                          arbitration clauses in legal documents. Features include document scanning, OCR,
                          real-time analysis, offline mode, AR visualization, and ML-powered risk assessment.
                       DESC

  spec.homepage         = 'https://github.com/arbitration-platform/ios-sdk'
  spec.license          = { :type => 'MIT', :file => 'LICENSE' }
  spec.author           = { 'Arbitration Platform' => 'sdk@arbitration-platform.com' }
  spec.source           = { :git => 'https://github.com/arbitration-platform/ios-sdk.git', :tag => spec.version.to_s }

  spec.ios.deployment_target = '15.0'
  spec.osx.deployment_target = '12.0'
  spec.tvos.deployment_target = '15.0'
  spec.watchos.deployment_target = '8.0'

  spec.swift_version = '5.9'
  spec.requires_arc = true

  # Main SDK
  spec.subspec 'Core' do |core|
    core.source_files = 'Sources/ArbitrationSDK/**/*.swift'
    core.resources = 'Sources/ArbitrationSDK/Resources/**/*'
    core.dependency 'Alamofire', '~> 5.8'
    core.dependency 'RealmSwift', '~> 10.45'
    core.frameworks = 'Foundation', 'UIKit', 'CoreML', 'Vision', 'NaturalLanguage'
  end

  # UI Components
  spec.subspec 'UI' do |ui|
    ui.source_files = 'Sources/ArbitrationUI/**/*.swift'
    ui.resources = 'Sources/ArbitrationUI/Resources/**/*'
    ui.dependency 'ArbitrationSDK/Core'
    ui.frameworks = 'SwiftUI', 'Combine'
  end

  # AR Capabilities
  spec.subspec 'AR' do |ar|
    ar.source_files = 'Sources/ArbitrationAR/**/*.swift'
    ar.dependency 'ArbitrationSDK/Core'
    ar.frameworks = 'ARKit', 'SceneKit', 'RealityKit'
    ar.ios.deployment_target = '15.0'
  end

  # Default subspecs
  spec.default_subspecs = 'Core', 'UI'

  # Testing
  spec.test_spec 'Tests' do |test_spec|
    test_spec.source_files = 'Tests/**/*.swift'
    test_spec.requires_app_host = true
  end

  # Build settings
  spec.pod_target_xcconfig = {
    'SWIFT_VERSION' => '5.9',
    'ENABLE_BITCODE' => 'NO',
    'OTHER_SWIFT_FLAGS' => '-Xfrontend -enable-experimental-concurrency'
  }
end