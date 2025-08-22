#import "OCRNativeModule.h"
#import <React/RCTLog.h>
#import <React/RCTUtils.h>
#import <CoreImage/CoreImage.h>

@implementation OCRNativeModule

RCT_EXPORT_MODULE();

+ (BOOL)requiresMainQueueSetup
{
    return NO;
}

- (NSArray<NSString *> *)supportedEvents
{
    return @[@"OCRProgress", @"OCRComplete", @"OCRError"];
}

RCT_EXPORT_METHOD(recognizeTextInImage:(NSString *)imagePath
                               resolver:(RCTPromiseResolveBlock)resolve
                               rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
            if (!image) {
                reject(@"IMAGE_ERROR", @"Failed to load image", nil);
                return;
            }
            
            [self performTextRecognition:image
                              completion:^(NSArray *results, NSError *error) {
                if (error) {
                    reject(@"OCR_ERROR", error.localizedDescription, error);
                } else {
                    resolve(@{
                        @"text": [self combineTextResults:results],
                        @"blocks": results,
                        @"confidence": [self calculateOverallConfidence:results]
                    });
                }
            }];
        }
        @catch (NSException *exception) {
            reject(@"OCR_EXCEPTION", exception.reason, nil);
        }
    });
}

RCT_EXPORT_METHOD(recognizeTextInImageWithBounds:(NSString *)imagePath
                                         resolver:(RCTPromiseResolveBlock)resolve
                                         rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
            if (!image) {
                reject(@"IMAGE_ERROR", @"Failed to load image", nil);
                return;
            }
            
            [self performTextRecognitionWithBounds:image
                                        completion:^(NSArray *results, NSError *error) {
                if (error) {
                    reject(@"OCR_ERROR", error.localizedDescription, error);
                } else {
                    resolve(@{
                        @"text": [self combineTextResults:results],
                        @"boundingBoxes": results,
                        @"confidence": [self calculateOverallConfidence:results]
                    });
                }
            }];
        }
        @catch (NSException *exception) {
            reject(@"OCR_EXCEPTION", exception.reason, nil);
        }
    });
}

RCT_EXPORT_METHOD(detectDocumentBounds:(NSString *)imagePath
                               resolver:(RCTPromiseResolveBlock)resolve
                               rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
            if (!image) {
                reject(@"IMAGE_ERROR", @"Failed to load image", nil);
                return;
            }
            
            [self detectDocumentInImage:image
                             completion:^(NSArray *corners, CGFloat confidence, NSError *error) {
                if (error) {
                    reject(@"DETECTION_ERROR", error.localizedDescription, error);
                } else {
                    resolve(@{
                        @"corners": corners ?: @[],
                        @"confidence": @(confidence)
                    });
                }
            }];
        }
        @catch (NSException *exception) {
            reject(@"DETECTION_EXCEPTION", exception.reason, nil);
        }
    });
}

RCT_EXPORT_METHOD(enhanceImageForOCR:(NSString *)imagePath
                             resolver:(RCTPromiseResolveBlock)resolve
                             rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
            if (!image) {
                reject(@"IMAGE_ERROR", @"Failed to load image", nil);
                return;
            }
            
            UIImage *enhancedImage = [self enhanceImage:image];
            NSString *outputPath = [self saveImageToTempFile:enhancedImage];
            
            if (outputPath) {
                resolve(@{@"path": outputPath});
            } else {
                reject(@"SAVE_ERROR", @"Failed to save enhanced image", nil);
            }
        }
        @catch (NSException *exception) {
            reject(@"ENHANCEMENT_EXCEPTION", exception.reason, nil);
        }
    });
}

RCT_EXPORT_METHOD(correctPerspective:(NSString *)imagePath
                             corners:(NSArray *)corners
                            resolver:(RCTPromiseResolveBlock)resolve
                            rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
            if (!image) {
                reject(@"IMAGE_ERROR", @"Failed to load image", nil);
                return;
            }
            
            UIImage *correctedImage = [self correctPerspectiveOfImage:image withCorners:corners];
            NSString *outputPath = [self saveImageToTempFile:correctedImage];
            
            if (outputPath) {
                resolve(@{@"path": outputPath});
            } else {
                reject(@"SAVE_ERROR", @"Failed to save corrected image", nil);
            }
        }
        @catch (NSException *exception) {
            reject(@"CORRECTION_EXCEPTION", exception.reason, nil);
        }
    });
}

#pragma mark - Text Recognition Implementation

- (void)performTextRecognition:(UIImage *)image
                    completion:(void (^)(NSArray *results, NSError *error))completion
{
    if (@available(iOS 13.0, *)) {
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
            if (error) {
                completion(nil, error);
                return;
            }
            
            NSMutableArray *results = [NSMutableArray array];
            for (VNRecognizedTextObservation *observation in request.results) {
                VNRecognizedText *recognizedText = [observation topCandidates:1].firstObject;
                if (recognizedText) {
                    [results addObject:@{
                        @"text": recognizedText.string,
                        @"confidence": @(recognizedText.confidence)
                    }];
                }
            }
            completion(results, nil);
        }];
        
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = YES;
        
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image.CGImage options:@{}];
        [handler performRequests:@[request] error:nil];
    } else {
        completion(nil, [NSError errorWithDomain:@"OCRNativeModule" code:1 userInfo:@{NSLocalizedDescriptionKey: @"Text recognition requires iOS 13.0 or later"}]);
    }
}

- (void)performTextRecognitionWithBounds:(UIImage *)image
                              completion:(void (^)(NSArray *results, NSError *error))completion
{
    if (@available(iOS 13.0, *)) {
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
            if (error) {
                completion(nil, error);
                return;
            }
            
            NSMutableArray *results = [NSMutableArray array];
            for (VNRecognizedTextObservation *observation in request.results) {
                VNRecognizedText *recognizedText = [observation topCandidates:1].firstObject;
                if (recognizedText) {
                    CGRect boundingBox = observation.boundingBox;
                    CGRect convertedBox = [self convertVisionRectToImageRect:boundingBox imageSize:image.size];
                    
                    [results addObject:@{
                        @"text": recognizedText.string,
                        @"confidence": @(recognizedText.confidence),
                        @"x": @(convertedBox.origin.x),
                        @"y": @(convertedBox.origin.y),
                        @"width": @(convertedBox.size.width),
                        @"height": @(convertedBox.size.height)
                    }];
                }
            }
            completion(results, nil);
        }];
        
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = YES;
        
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image.CGImage options:@{}];
        [handler performRequests:@[request] error:nil];
    } else {
        completion(nil, [NSError errorWithDomain:@"OCRNativeModule" code:1 userInfo:@{NSLocalizedDescriptionKey: @"Text recognition requires iOS 13.0 or later"}]);
    }
}

- (void)detectDocumentInImage:(UIImage *)image
                   completion:(void (^)(NSArray *corners, CGFloat confidence, NSError *error))completion
{
    VNDetectRectanglesRequest *request = [[VNDetectRectanglesRequest alloc] initWithCompletionHandler:^(VNRequest *request, NSError *error) {
        if (error) {
            completion(nil, 0.0, error);
            return;
        }
        
        VNRectangleObservation *observation = request.results.firstObject;
        if (observation) {
            NSArray *corners = @[
                @{@"x": @(observation.topLeft.x), @"y": @(observation.topLeft.y)},
                @{@"x": @(observation.topRight.x), @"y": @(observation.topRight.y)},
                @{@"x": @(observation.bottomRight.x), @"y": @(observation.bottomRight.y)},
                @{@"x": @(observation.bottomLeft.x), @"y": @(observation.bottomLeft.y)}
            ];
            completion(corners, observation.confidence, nil);
        } else {
            completion(nil, 0.0, nil);
        }
    }];
    
    request.minimumAspectRatio = 0.3;
    request.maximumAspectRatio = 1.0;
    request.minimumSize = 0.1;
    request.maximumObservations = 1;
    
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image.CGImage options:@{}];
    [handler performRequests:@[request] error:nil];
}

#pragma mark - Image Processing Implementation

- (UIImage *)enhanceImage:(UIImage *)image
{
    CIImage *ciImage = [CIImage imageWithCGImage:image.CGImage];
    CIContext *context = [CIContext contextWithOptions:nil];
    
    // Apply filters to enhance text readability
    
    // 1. Adjust exposure and brightness
    CIFilter *exposureFilter = [CIFilter filterWithName:@"CIExposureAdjust"];
    [exposureFilter setValue:ciImage forKey:kCIInputImageKey];
    [exposureFilter setValue:@(0.3) forKey:kCIInputEVKey];
    ciImage = exposureFilter.outputImage;
    
    // 2. Increase contrast
    CIFilter *contrastFilter = [CIFilter filterWithName:@"CIColorControls"];
    [contrastFilter setValue:ciImage forKey:kCIInputImageKey];
    [contrastFilter setValue:@(1.2) forKey:kCIInputContrastKey];
    ciImage = contrastFilter.outputImage;
    
    // 3. Sharpen the image
    CIFilter *sharpenFilter = [CIFilter filterWithName:@"CISharpenLuminance"];
    [sharpenFilter setValue:ciImage forKey:kCIInputImageKey];
    [sharpenFilter setValue:@(0.8) forKey:kCIInputSharpnessKey];
    ciImage = sharpenFilter.outputImage;
    
    // 4. Reduce noise
    CIFilter *noiseFilter = [CIFilter filterWithName:@"CINoiseReduction"];
    [noiseFilter setValue:ciImage forKey:kCIInputImageKey];
    [noiseFilter setValue:@(0.02) forKey:kCIInputNoiseReductionKey];
    [noiseFilter setValue:@(0.4) forKey:kCIInputSharpnessKey];
    ciImage = noiseFilter.outputImage;
    
    CGImageRef cgImage = [context createCGImage:ciImage fromRect:ciImage.extent];
    UIImage *enhancedImage = [UIImage imageWithCGImage:cgImage];
    CGImageRelease(cgImage);
    
    return enhancedImage;
}

- (UIImage *)correctPerspectiveOfImage:(UIImage *)image withCorners:(NSArray *)corners
{
    if (corners.count != 4) {
        return image;
    }
    
    CIImage *ciImage = [CIImage imageWithCGImage:image.CGImage];
    CIFilter *perspectiveFilter = [CIFilter filterWithName:@"CIPerspectiveCorrection"];
    
    // Convert corner points to CIVector format
    NSDictionary *topLeft = corners[0];
    NSDictionary *topRight = corners[1];
    NSDictionary *bottomRight = corners[2];
    NSDictionary *bottomLeft = corners[3];
    
    [perspectiveFilter setValue:ciImage forKey:kCIInputImageKey];
    [perspectiveFilter setValue:[CIVector vectorWithX:[topLeft[@"x"] floatValue] Y:[topLeft[@"y"] floatValue]] forKey:@"inputTopLeft"];
    [perspectiveFilter setValue:[CIVector vectorWithX:[topRight[@"x"] floatValue] Y:[topRight[@"y"] floatValue]] forKey:@"inputTopRight"];
    [perspectiveFilter setValue:[CIVector vectorWithX:[bottomRight[@"x"] floatValue] Y:[bottomRight[@"y"] floatValue]] forKey:@"inputBottomRight"];
    [perspectiveFilter setValue:[CIVector vectorWithX:[bottomLeft[@"x"] floatValue] Y:[bottomLeft[@"y"] floatValue]] forKey:@"inputBottomLeft"];
    
    CIImage *correctedImage = perspectiveFilter.outputImage;
    CIContext *context = [CIContext contextWithOptions:nil];
    CGImageRef cgImage = [context createCGImage:correctedImage fromRect:correctedImage.extent];
    UIImage *resultImage = [UIImage imageWithCGImage:cgImage];
    CGImageRelease(cgImage);
    
    return resultImage;
}

#pragma mark - Utility Methods

- (NSString *)combineTextResults:(NSArray *)results
{
    NSMutableArray *textParts = [NSMutableArray array];
    for (NSDictionary *result in results) {
        NSString *text = result[@"text"];
        if (text && text.length > 0) {
            [textParts addObject:text];
        }
    }
    return [textParts componentsJoinedByString:@"\n"];
}

- (NSNumber *)calculateOverallConfidence:(NSArray *)results
{
    if (results.count == 0) {
        return @(0.0);
    }
    
    CGFloat totalConfidence = 0.0;
    for (NSDictionary *result in results) {
        NSNumber *confidence = result[@"confidence"];
        if (confidence) {
            totalConfidence += confidence.floatValue;
        }
    }
    
    return @(totalConfidence / results.count);
}

- (CGRect)convertVisionRectToImageRect:(CGRect)visionRect imageSize:(CGSize)imageSize
{
    // Vision framework uses normalized coordinates (0-1) with origin at bottom-left
    // Convert to UIImage coordinates with origin at top-left
    CGFloat x = visionRect.origin.x * imageSize.width;
    CGFloat y = (1.0 - visionRect.origin.y - visionRect.size.height) * imageSize.height;
    CGFloat width = visionRect.size.width * imageSize.width;
    CGFloat height = visionRect.size.height * imageSize.height;
    
    return CGRectMake(x, y, width, height);
}

- (NSString *)saveImageToTempFile:(UIImage *)image
{
    NSData *imageData = UIImageJPEGRepresentation(image, 0.9);
    if (!imageData) {
        return nil;
    }
    
    NSString *tempDir = NSTemporaryDirectory();
    NSString *fileName = [NSString stringWithFormat:@"processed_image_%@.jpg", [[NSUUID UUID] UUIDString]];
    NSString *filePath = [tempDir stringByAppendingPathComponent:fileName];
    
    BOOL success = [imageData writeToFile:filePath atomically:YES];
    return success ? filePath : nil;
}

@end