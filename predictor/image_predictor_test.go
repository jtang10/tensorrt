package predictor

import (
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/rai-project/dlframework/framework/options"
	raiimage "github.com/rai-project/image"
	"github.com/rai-project/image/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	trt "github.com/rai-project/tensorrt"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

func normalizeImageCHW(in0 image.Image, mean []float32, scale []float32) ([]float32, error) {
	height := in0.Bounds().Dy()
	width := in0.Bounds().Dx()
	out := make([]float32, 3*height*width)
	switch in := in0.(type) {
	case *types.RGBImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				rgb := in.Pix[offset : offset+3]
				r, g, b := rgb[0], rgb[1], rgb[2]
				out[0*width*height+y*width+x] = (float32(r) - mean[0]) / scale[0]
				out[1*width*height+y*width+x] = (float32(g) - mean[1]) / scale[1]
				out[2*width*height+y*width+x] = (float32(b) - mean[2]) / scale[2]
			}
		}
	case *types.BGRImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				bgr := in.Pix[offset : offset+3]
				b, g, r := bgr[0], bgr[1], bgr[2]
				out[0*width*height+y*width+x] = (float32(b) - mean[0]) / scale[0]
				out[1*width*height+y*width+x] = (float32(g) - mean[1]) / scale[1]
				out[2*width*height+y*width+x] = (float32(r) - mean[2]) / scale[2]
			}
		}
	default:
		panic("unreachable")
	}

	return out, nil
}

func TestNewImageClassificationPredictor(t *testing.T) {
	trt.Register()
	model, err := trt.FrameworkManifest.FindModel("vgg19:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	predictor, err := NewImageClassificationPredictor(*model)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)

	defer predictor.Close()

	imgPredictor, ok := predictor.(*ImageClassificationPredictor)
	assert.True(t, ok)

	assert.NotEmpty(t, imgPredictor)
}

func TestImageClassification(t *testing.T) {
	trt.Register()
	model, err := trt.FrameworkManifest.FindModel("vgg19:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewImageClassificationPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "platypus.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	channels := preprocessOpts.Dims[0]
	height := preprocessOpts.Dims[1]
	width := preprocessOpts.Dims[2]
	mode := preprocessOpts.ColorMode
	fmt.Println("dims:", preprocessOpts.Dims)
	fmt.Println("mode:", mode)

	imgOpts := []raiimage.Option{
		raiimage.Mode(mode),
		raiimage.Width(width),
		raiimage.Height(height),
		raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear),
	}

	if mode == types.RGBMode {
		imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	} else {
		imgOpts = append(imgOpts, raiimage.Mode(types.BGRMode))
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, raiimage.Resized(height, width))
	imgOpts = append(imgOpts, raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := raiimage.Resize(img, imgOpts...)

	input := make([]*gotensor.Dense, batchSize)
	imgFloats, err := normalizeImageCHW(resized, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for ii := 0; ii < batchSize; ii++ {
		input[ii] = gotensor.New(
			gotensor.WithShape(height, width, channels),
			gotensor.WithBacking(imgFloats),
		)
	}

	err = predictor.Predict(ctx, input)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}
	fmt.Println(pred[0][0].GetClassification().GetIndex(), ": ", pred[0][0].GetProbability())
	assert.InDelta(t, float32(0.99999), pred[0][0].GetProbability(), 0.001)
	assert.Equal(t, int32(103), pred[0][0].GetClassification().GetIndex())
}

// func TestInstanceSegmentation(t *testing.T) {
// 	trt.Register()
// 	model, err := trt.FrameworkManifest.FindModel("mask_rcnn_inception_v2_coco:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 1
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewInstanceSegmentationPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "lane_control.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}
// 	img, err := raiimage.Read(r)
// 	if err != nil {
// 		panic(err)
// 	}

// 	height := img.Bounds().Dy()
// 	width := img.Bounds().Dx()
// 	channels := 3
// 	input := make([]*gotensor.Dense, batchSize)
// 	imgBytes := img.(*types.RGBImage).Pix

// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(height, width, channels),
// 			gotensor.WithBacking(imgBytes),
// 		)
// 	}

// 	err = predictor.Predict(ctx, input)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	assert.InDelta(t, float32(0.998607), pred[0][0].GetProbability(), 0.001)
// }

// func TestObjectDetection(t *testing.T) {
// 	trt.Register()
// 	model, err := trt.FrameworkManifest.FindModel("SSD_300_VGG16_Atrous_COCO:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 2
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewObjectDetectionPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	preprocessOpts, err := predictor.GetPreprocessOptions()
// 	assert.NoError(t, err)
// 	channels := preprocessOpts.Dims[0]
// 	height := preprocessOpts.Dims[1]
// 	width := preprocessOpts.Dims[2]
// 	mode := preprocessOpts.ColorMode

// 	imgOpts := []raiimage.Option{
// 		raiimage.Mode(mode),
// 		raiimage.Width(width),
// 		raiimage.Height(height),
// 		raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear),
// 	}

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "3dogs.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}

// 	img, err := raiimage.Read(r, imgOpts...)
// 	if err != nil {
// 		panic(err)
// 	}

// 	input := make([]*gotensor.Dense, batchSize)
// 	imgFloats, err := normalizeImageCHW(img, preprocessOpts.MeanImage, preprocessOpts.Scale)
// 	if err != nil {
// 		panic(err)
// 	}

// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(height, width, channels),
// 			gotensor.WithBacking(imgFloats),
// 		)
// 	}

// 	err = predictor.Predict(ctx, input)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}
// 	assert.InDelta(t, float32(0.996272), pred[0][0].GetProbability(), 0.001)
// 	assert.Equal(t, int32(11), pred[0][0].GetBoundingBox().GetIndex())
// }

// func max(x, y int) int {
// 	if x < y {
// 		return y
// 	}
// 	return x
// }

// func TestSemanticSegmentation(t *testing.T) {
// 	trt.Register()
// 	model, err := trt.FrameworkManifest.FindModel("DeepLabv3_PASCAL_VOC_Train_Aug:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 1
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewSemanticSegmentationPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "lane_control.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}
// 	img, err := raiimage.Read(r)
// 	if err != nil {
// 		panic(err)
// 	}

// 	height := img.Bounds().Dy()
// 	width := img.Bounds().Dx()
// 	channels := 3
// 	inputSize := 513
// 	resizeRatio := float32(inputSize) / float32(max(width, height))
// 	targetWidth := int(resizeRatio * float32(width))
// 	targetHeight := int(resizeRatio * float32(height))
// 	resized, err := raiimage.Resize(img, raiimage.Resized(targetHeight, targetWidth))
// 	if err != nil {
// 		panic(err)
// 	}
// 	input := make([]*gotensor.Dense, batchSize)
// 	imgBytes := resized.(*types.RGBImage).Pix
// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(targetHeight, targetWidth, channels),
// 			gotensor.WithBacking(imgBytes),
// 		)
// 	}

// 	err = predictor.Predict(ctx, input)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	sseg := pred[0][0].GetSemanticSegment()
// 	intMask := sseg.GetIntMask()

// 	assert.Equal(t, int32(7), intMask[72122])
// }
