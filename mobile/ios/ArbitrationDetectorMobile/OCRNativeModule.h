#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>
#import <Vision/Vision.h>
#import <UIKit/UIKit.h>

@interface OCRNativeModule : RCTEventEmitter <RCTBridgeModule>

// Text recognition methods
- (void)recognizeTextInImage:(NSString *)imagePath
                    resolver:(RCTPromiseResolveBlock)resolve
                    rejecter:(RCTPromiseRejectBlock)reject;

- (void)recognizeTextInImageWithBounds:(NSString *)imagePath
                              resolver:(RCTPromiseResolveBlock)resolve
                              rejecter:(RCTPromiseRejectBlock)reject;

// Document detection methods
- (void)detectDocumentBounds:(NSString *)imagePath
                    resolver:(RCTPromiseResolveBlock)resolve
                    rejecter:(RCTPromiseRejectBlock)reject;

// Image processing methods
- (void)enhanceImageForOCR:(NSString *)imagePath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject;

- (void)correctPerspective:(NSString *)imagePath
                  corners:(NSArray *)corners
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject;

@end