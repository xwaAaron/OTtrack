package com.example.objecttrack;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;


public class Track_tools {

    private static final String TAG = "CameraActivity";

    public boolean m_use_scale=true;
    public boolean m_use_color=true;
    public boolean m_use_subpixel_localization=true;
    public boolean m_use_subgrid_scale=true;
    public boolean m_use_multithreading=true;
    public boolean m_use_cnfeat =false;
    public boolean m_use_linearkernel =false;


    BBox_c p_pose=new BBox_c();
    boolean p_resize_image = false;

    double p_padding = 1.5;
    double p_output_sigma_factor = 0.1;
    double p_output_sigma;
    double p_kernel_sigma = 0.5;    //def = 0.5
    double p_lambda = 1e-4;         //regularization in learning step
    double p_interp_factor = 0.02;  //def = 0.02, linear interpolation factor for adaptation
    int p_cell_size = 4;            //4 for hog (= bin_size)
    int[] p_windows_size=new int[2];
    Mat p_cos_window;
    int p_num_scales=7;
    double p_scale_step = 1.02;
    double p_current_scale = 1.;
    double[] p_min_max_scale=new double[2];
    ArrayList<Double> p_scales=new ArrayList<Double>();

    //model
    ComplexMat p_yf;
    ComplexMat p_model_alphaf;
    ComplexMat p_model_alphaf_num;
    ComplexMat p_model_alphaf_den;
    ComplexMat p_model_xf;

    /*
    padding             ... extra area surrounding the target           (1.5)
    kernel_sigma        ... gaussian kernel bandwidth                   (0.5)
    lambda              ... regularization                              (1e-4)
    interp_factor       ... linear interpolation factor for adaptation  (0.02)
    output_sigma_factor ... spatial bandwidth (proportional to target)  (0.1)
    cell_size           ... hog cell size                               (4)
    */
    Track_tools(double padding, double kernel_sigma, double lambda, double interp_factor, double output_sigma_factor, int cell_size)
    {
        p_padding=padding;
        p_kernel_sigma=kernel_sigma;
        p_lambda=lambda;
        p_interp_factor=interp_factor;
        p_output_sigma_factor=output_sigma_factor;
        p_cell_size=cell_size;
    }
    Track_tools(){};
    // input the boundingbox of target and the whole image
    void init(Mat img, Rect bbox)
    {
        double x1 = bbox.x, x2 = bbox.x + bbox.width, y1 = bbox.y, y2 = bbox.y + bbox.height;

        //modify boraderline  the bounding box is too large
        if (x1 < 0) x1 = 0.;
        if (x2 > img.cols()-1) x2 = img.cols() - 1;
        if (y1 < 0) y1 = 0;
        if (y2 > img.rows()-1) y2 = img.rows() - 1;
        //the bouding box is too small
        if (x2-x1 < 2*p_cell_size) {
            double diff = (2*p_cell_size -x2+x1)/2.;
            if (x1 - diff >= 0 && x2 + diff < img.cols()){
                x1 -= diff;
                x2 += diff;
            } else if (x1 - 2*diff >= 0) {
                x1 -= 2*diff;
            } else {
                x2 += 2*diff;
            }
        }
        if (y2-y1 < 2*p_cell_size) {
            double diff = (2*p_cell_size -y2+y1)/2.;
            if (y1 - diff >= 0 && y2 + diff < img.rows()){
                y1 -= diff;
                y2 += diff;
            } else if (y1 - 2*diff >= 0) {
                y1 -= 2*diff;
            } else {
                y2 += 2*diff;
            }
        }
        //modify the target box
        p_pose.w = x2-x1;
        p_pose.h = y2-y1;
        p_pose.cx = x1 + p_pose.w/2.;
        p_pose.cy = y1 + p_pose.h/2.;

        //convert the input image to gray image
        Mat input_gray=img.clone(), input_rgb = img.clone();
        if (img.channels() == 3){
            Imgproc.cvtColor(img, input_gray, Imgproc.COLOR_BGR2GRAY);
            input_gray.convertTo(input_gray, CvType.CV_32FC1);
        }else
            img.convertTo(input_gray, CvType.CV_32FC1);

        // if the target image is too large
        if (p_pose.w * p_pose.h > 100.*100.) {
            //std::cout << "resizing image by factor of 2" << std::endl;
            p_resize_image = true;
            p_pose.scale(0.5);
            Imgproc.resize(input_gray, input_gray, new Size(0,0), 0.5, 0.5, Imgproc.INTER_AREA);
            Imgproc.resize(input_rgb, input_rgb, new Size(0,0), 0.5, 0.5, Imgproc.INTER_AREA);
        }

        //compute win size + fit to fhog cell size
        p_windows_size[0] = (int) Math.round(p_pose.w * (1. + p_padding) / p_cell_size) * p_cell_size;    //width
        p_windows_size[1] = (int) Math.round(p_pose.h * (1. + p_padding) / p_cell_size) * p_cell_size;   //height
        //whether to use multi scales
        if (m_use_scale)
            for (int i = -p_num_scales/2; i <= p_num_scales/2; ++i)
                p_scales.add(Math.pow(p_scale_step, i));
        else
            p_scales.add(1.);

        p_current_scale = 1.;

        double min_size_ratio = Math.max(5.*p_cell_size/p_windows_size[0], 5.*p_cell_size/p_windows_size[1]);
        double max_size_ratio = Math.min(Math.floor((img.cols() + p_windows_size[0]/3)/p_cell_size)*p_cell_size/p_windows_size[0], Math.floor((img.rows() + p_windows_size[1]/3)/p_cell_size)*p_cell_size/p_windows_size[1]);
        p_min_max_scale[0] = Math.pow(p_scale_step, Math.ceil(Math.log(min_size_ratio) / Math.log(p_scale_step)));
        p_min_max_scale[1] = Math.pow(p_scale_step, Math.floor(Math.log(max_size_ratio) / Math.log(p_scale_step)));

        String s;
        s="init: img size "+img.cols()+"  "+img.rows();
        Log.i(TAG,s);
        s="init: window size. "+p_windows_size[0] + " " + p_windows_size[1];
        Log.i(TAG,s);
        s="init: min max scales factors: " + p_min_max_scale[0] + " " + p_min_max_scale[1];
        Log.i(TAG,s);

        //
        p_output_sigma = Math.sqrt(p_pose.w*p_pose.h) * p_output_sigma_factor / (double)(p_cell_size);

        //window weights, i.e. labels   obtain a gaussian template and add a hanning window
        p_yf = fft2(gaussian_shaped_labels(p_output_sigma, p_windows_size[0]/p_cell_size, p_windows_size[1]/p_cell_size));   //obtain a fixed size guassian kernel
        p_cos_window = cosine_window_function(p_yf.cols, p_yf.rows);

        //obtain a sub-window for training initial model
        ArrayList<Mat> path_feat = get_features(input_rgb, input_gray, (int)p_pose.cx, (int)p_pose.cy, p_windows_size[0], p_windows_size[1],1);
        p_model_xf = fft2(path_feat, p_cos_window);    // fft the feather mat with a hanning window in order to erase the bouding effect

        if (m_use_linearkernel) {
            ComplexMat xfconj = p_model_xf.conj();
            p_model_alphaf_num = xfconj.mul(p_yf);
            p_model_alphaf_den = (p_model_xf.muln(xfconj));
        } else {
            //Kernel Ridge Regression, calculate alphas (in Fourier domain)
            ComplexMat kf = gaussian_correlation(p_model_xf, p_model_xf, p_kernel_sigma, true);
            p_model_alphaf_num = p_yf.muln(kf);
            p_model_alphaf_den = kf.muln(kf.Addconstant(p_lambda));
        }
        p_model_alphaf = p_model_alphaf_num.div(p_model_alphaf_den);
//        p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training
    }

    ComplexMat fft2(Mat input)
    {
        Mat complex_result=new Mat();
        //Mat padded=new Mat();                            //expand input image to optimal size
        //int m = Core.getOptimalDFTSize( input.rows() );
        //int n = Core.getOptimalDFTSize( input.cols() ); // on the border add zero pixels
        //Core.copyMakeBorder(input, padded, 0, m - input.rows(), 0, n - input.cols(), Core.BORDER_CONSTANT, Scalar.all(0));
        //return ComplexMat(complex_result(cv::Range(0, input.rows), cv::Range(0, input.cols)));
        Core.dft(input, complex_result, Core.DFT_COMPLEX_OUTPUT,0);
        return new ComplexMat(complex_result);
    }
    ComplexMat fft2(ArrayList<Mat> input, Mat cos_window)
    {
        int n_channels = input.size();
        Mat m=input.get(0);
        ComplexMat result=new ComplexMat(m.rows(), m.cols(), n_channels);
        for (int i = 0; i < n_channels; ++i){
            Mat complex_result=new Mat();
//        cv::Mat padded;                            //expand input image to optimal size
//        int m = cv::getOptimalDFTSize( input[0].rows );
//        int n = cv::getOptimalDFTSize( input[0].cols ); // on the border add zero pixels

//        copyMakeBorder(input[i].mul(cos_window), padded, 0, m - input[0].rows, 0, n - input[0].cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//        cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//        result.set_channel(i, complex_result(cv::Range(0, input[0].rows), cv::Range(0, input[0].cols)));
            Mat temp=input.get(i);
            Core.dft(temp.mul(cos_window), complex_result, Core.DFT_COMPLEX_OUTPUT,0);
            result.set_channel(i, complex_result);
        }
        return result;
    }

    Mat cosine_window_function(int dim1, int dim2)
    {
        Mat m1=new Mat(1, dim1, CvType.CV_32FC1);
        Mat m2=new Mat(dim2, 1, CvType.CV_32FC1);
        double N_inv = 1./((double)(dim1)-1.);
        for (int i = 0; i < dim1; ++i)
        {
            double a=0.5*(1. - Math.cos(2. * Math.PI * (double)(i) * N_inv));
            m1.put(0,i,a);
        }
        N_inv = 1./((double)(dim2)-1.);
        for (int i = 0; i < dim2; ++i)
        {
            double a = 0.5*(1. - Math.cos(2. * Math.PI * (double)(i) * N_inv));
            m2.put(i,0,a);
        }
        Mat ret = new Mat(dim2, dim1, CvType.CV_32FC1);
        Core.gemm(m2,m1,1,new Mat(),0,ret,0);
        return ret;
    }



//   Gaussian-shaped labels for all shifts of a sample.
//   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
//   Creates an array of labels (regression targets) for all shifts of a
//   sample of dimensions SZ. The output will have size SZ, representing
//   one label for each possible shift. The labels will be Gaussian-shaped,
//   with the peak at 0-shift (top-left element of the array), decaying
//   as the distance increases, and wrapping around at the borders.
//   The Gaussian function has spatial bandwidth SIGMA.
    Mat gaussian_shaped_labels(double sigma, int dim1, int dim2)
    {
        Mat labels=new Mat(dim2, dim1, CvType.CV_32FC1);
        int[] range_y = {-dim2 / 2, dim2 - dim2 / 2};
        int[] range_x = {-dim1 / 2, dim1 - dim1 / 2};

        double sigma_s = sigma*sigma;

        for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j){
            double y_s = y*y;
            for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i){
                double temp= Math.exp(-0.5 * (y_s + x*x) / sigma_s);
                labels.put(j,i,temp);
            }
        }
        //rotate so that 1 is at top-left corner (see KCF paper for explanation)
        Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
        //sanity check, 1 at top left corner
        assert(rot_labels.get(0,0)[0] >= 1.f - 1e-10f);
        return rot_labels;
    }

    Mat circshift(Mat patch, int x_rot, int y_rot)
    {
        Mat rot_patch=new Mat(patch.size(), CvType.CV_32FC1);
        Mat tmp_x_rot=new Mat(patch.size(), CvType.CV_32FC1);

        //circular rotate x-axis
        if (x_rot < 0) {
            //move part that does not rotate over the edge
            Range orig_range=new Range(-x_rot, patch.cols());
            Range rot_range=new Range(0, patch.cols() - (-x_rot));
            patch.submat(Range.all(),orig_range).copyTo(tmp_x_rot.submat(Range.all(),rot_range));

            //rotated part
            orig_range = new Range(0, -x_rot);
            rot_range = new Range(patch.cols() - (-x_rot), patch.cols());
            patch.submat(Range.all(),orig_range).copyTo(tmp_x_rot.submat(Range.all(),rot_range));
        }else if (x_rot > 0){
            //move part that does not rotate over the edge
            Range orig_range=new Range(0, patch.cols() - x_rot);
            Range rot_range=new Range(x_rot, patch.cols());
            patch.submat(Range.all(), orig_range).copyTo(tmp_x_rot.submat(Range.all(), rot_range));

            //rotated part
            orig_range = new Range(patch.cols() - x_rot, patch.cols());
            rot_range = new Range(0, x_rot);
            patch.submat(Range.all(), orig_range).copyTo(tmp_x_rot.submat(Range.all(), rot_range));
        }else {    //zero rotation
            //move part that does not rotate over the edge
            Range orig_range=new Range(0, patch.cols());
            Range rot_range=new Range(0, patch.cols());
            patch.submat(Range.all(), orig_range).copyTo(tmp_x_rot.submat(Range.all(), rot_range));
        }

        //circular rotate y-axis
        if (y_rot < 0) {
            //move part that does not rotate over the edge
            Range orig_range=new Range(-y_rot, patch.rows());
            Range rot_range=new Range(0, patch.rows() - (-y_rot));
            tmp_x_rot.submat(orig_range, Range.all()).copyTo(rot_patch.submat(rot_range, Range.all()));

            //rotated part
            orig_range = new Range(0, -y_rot);
            rot_range = new Range(patch.rows() - (-y_rot), patch.rows());
            tmp_x_rot.submat(orig_range, Range.all()).copyTo(rot_patch.submat(rot_range, Range.all()));
        }else if (y_rot > 0){
            //move part that does not rotate over the edge
            Range orig_range=new Range(0, patch.rows() - y_rot);
            Range rot_range=new Range(y_rot, patch.rows());
            tmp_x_rot.submat(orig_range, Range.all()).copyTo(rot_patch.submat(rot_range, Range.all()));

            //rotated part
            orig_range = new Range(patch.rows() - y_rot, patch.rows());
            rot_range = new Range(0, y_rot);
            tmp_x_rot.submat(orig_range, Range.all()).copyTo(rot_patch.submat(rot_range, Range.all()));
        }else { //zero rotation
            //move part that does not rotate over the edge
            Range orig_range=new Range(0, patch.rows());
            Range rot_range=new Range(0, patch.rows());
            tmp_x_rot.submat(orig_range, Range.all()).copyTo(rot_patch.submat(rot_range, Range.all()));
        }

        return rot_patch;
    }

    ArrayList<Mat> get_features(Mat input_rgb, Mat input_gray, int cx, int cy, int size_x, int size_y, double scale)
    {
        int size_x_scaled = (int) Math.floor(size_x*scale);
        int size_y_scaled = (int) Math.floor(size_y*scale);

        Mat patch_gray = get_subwindow(input_gray, cx, cy, size_x_scaled, size_y_scaled);
        Mat patch_rgb = get_subwindow(input_rgb, cx, cy, size_x_scaled, size_y_scaled);

        //resize to default size
        if (scale > 1.){
            //if we downsample use  INTER_AREA interpolation
            Imgproc.resize(patch_gray, patch_gray, new Size(size_x, size_y), 0., 0., Imgproc.INTER_AREA);
        }else {
            Imgproc.resize(patch_gray, patch_gray, new Size(size_x, size_y), 0., 0., Imgproc.INTER_LINEAR);
        }

        // get hog features
       // ArrayList<Mat> hog_feat = FHoG.extract(patch_gray);

        //get color rgb features (simple r,g,b channels)
        ArrayList<Mat> color_feat=new ArrayList<Mat>();
        if ((m_use_color || m_use_cnfeat) && input_rgb.channels() == 3) {
            //resize to default size
            if (scale > 1.){
                //if we downsample use  INTER_AREA interpolation
                Imgproc.resize(patch_rgb, patch_rgb, new Size(size_x/p_cell_size, size_y/p_cell_size), 0., 0., Imgproc.INTER_AREA);
            }else {
                Imgproc.resize(patch_rgb, patch_rgb, new Size(size_x/p_cell_size, size_y/p_cell_size), 0., 0., Imgproc.INTER_LINEAR);
            }
        }

        if (m_use_color && input_rgb.channels() == 3) {
            //use rgb color space
            Mat patch_rgb_norm=new Mat();
            patch_rgb.convertTo(patch_rgb_norm, CvType.CV_32F, 1. / 255., -0.5);
            Mat ch1=new Mat(patch_rgb_norm.size(), CvType.CV_32FC1);
            Mat ch2=new Mat(patch_rgb_norm.size(), CvType.CV_32FC1);
            Mat ch3=new Mat(patch_rgb_norm.size(), CvType.CV_32FC1);
            ArrayList<Mat> rgb = new ArrayList<Mat>();
            rgb.add(ch1);
            rgb.add(ch2);
            rgb.add(ch3);
            Core.split(patch_rgb_norm, rgb);
            color_feat.addAll(color_feat.size(), rgb);
        }
/*
        if (m_use_cnfeat && input_rgb.channels() == 3) {
            ArrayList<Mat> cn_feat = CNFeat::extract(patch_rgb);
            color_feat.insert(color_feat.end(), cn_feat.begin(), cn_feat.end());
        }
*/
        //hog_feat.addAll(hog_feat.size(),color_feat);
        return new ArrayList<>();
    }

    // Returns sub-window of image input centered at [cx, cy] coordinates),
    // with size [width, height]. If any pixels are outside of the image,
    //they will replicate the values at the borders.
    Mat get_subwindow(Mat input, int cx, int cy, int width, int height)
    {
        Mat patch=new Mat();

        int x1 = cx - width/2;
        int y1 = cy - height/2;
        int x2 = cx + width/2;
        int y2 = cy + height/2;

        //out of image
        if (x1 >= input.cols() || y1 >= input.rows() || x2 < 0 || y2 < 0) {
            patch.create(height, width, input.type());
            patch.setTo(new Scalar(0.f));
            return patch;
        }

        int top = 0, bottom = 0, left = 0, right = 0;

        //fit to image coordinates, set border extensions;
        if (x1 < 0) {
            left = -x1;
            x1 = 0;
        }
        if (y1 < 0) {
            top = -y1;
            y1 = 0;
        }
        if (x2 >= input.cols()) {
            right = x2 - input.cols() + width % 2;
            x2 = input.cols();
        } else
            x2 += width % 2;

        if (y2 >= input.rows()) {
            bottom = y2 - input.rows() + height % 2;
            y2 = input.rows();
        } else
            y2 += height % 2;

        //patch=input.submat(new Rect(x1,y1,width,height));
        if (x2 - x1 == 0 || y2 - y1 == 0)
            patch = Mat.zeros(height, width, CvType.CV_32FC1);
        else
            Core.copyMakeBorder(input.submat(new Range(y1, y2), new Range(x1, x2)), patch, top, bottom, left, right, Core.BORDER_REPLICATE);
        //使用Opencv来复制图像并填充边界
        //sanity check
        assert(patch.cols() == width && patch.rows() == height);
        return patch;
    }

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative
    // shifts between input images X and Y, which must both be MxN. They must
    // also be periodic (ie., pre-processed with a cosine window). The result
    // is an MxN map of responses.
    // Inputs and output are all in the Fourier domain.
    ComplexMat gaussian_correlation(ComplexMat xf, ComplexMat  yf, double sigma, boolean auto_correlation)
    {
        float xf_sqr_norm = xf.sqr_norm();
        float yf_sqr_norm = auto_correlation ? xf_sqr_norm : yf.sqr_norm();

        ComplexMat xyf = auto_correlation ? xf.sqr_mag() : xf.muln(yf.conj());

        //ifft2 and sum over 3rd dimension, we dont care about individual channels
        Mat xy_sum=new Mat(xf.rows, xf.cols, CvType.CV_32FC1);
        xy_sum.setTo(new Scalar(0));
        Mat ifft2_res = ifft2(xyf);
        int channels=ifft2_res.channels();
        for (int y = 0; y < xf.rows; ++y) {
            Mat ifft2_row=ifft2_res.row(y);
            for (int x = 0; x < xf.cols; ++x){
                float[] f=new float[channels];
                ifft2_res.get(0,x,f);
                float sum=0;
                for(int k=0;k<channels;k++)
                {
                    sum+=f[k];
                }
                xy_sum.put(y,x,sum);
            }
        }

        float numel_xf_inv = 1.f/(xf.cols * xf.rows * xf.n_channels);
        Mat tmp=new Mat();
        Mat t1=new Mat(xf.rows, xf.cols, CvType.CV_32FC1,new Scalar(2));
        Mat t2=new Mat(xf.rows, xf.cols, CvType.CV_32FC1,new Scalar(xf_sqr_norm + yf_sqr_norm));
        Mat t3=xy_sum.mul(t1);
        Mat t4=new Mat(xf.rows, xf.cols, CvType.CV_32FC1);
        Core.subtract(t2,t3,t4);
        t1=new Mat(xf.rows, xf.cols, CvType.CV_32FC1,new Scalar(numel_xf_inv));
        t2=t4.mul(t1);
        Core.max(t2, new Scalar(0),t3);
        double x=- 1.f / (sigma * sigma);
        t4=t3.mul(new Mat(xf.rows, xf.cols, CvType.CV_32FC1,new Scalar(x)));
        Core.exp(t4, tmp);
        return fft2(tmp);
    }

    Mat ifft2(ComplexMat inputf)
    {
        Mat real_result=new Mat();
        if (inputf.n_channels == 1){
            Core.dft(inputf.to_cv_mat(), real_result, Core.DFT_INVERSE | Core.DFT_REAL_OUTPUT | Core.DFT_SCALE,0);
        } else {
            ArrayList<Mat> mat_channels = inputf.to_cv_mat_vector();
            ArrayList<Mat> ifft_mats=new ArrayList<Mat>(inputf.n_channels);
            for (int i = 0; i < inputf.n_channels; ++i) {
                Core.dft(mat_channels.get(i), ifft_mats.get(i), Core.DFT_INVERSE | Core.DFT_REAL_OUTPUT | Core.DFT_SCALE,0);
            }
            Core.merge(ifft_mats, real_result);
        }
        return real_result;
    }


    void setTrackerPose(BBox_c  bbox, Mat img)
    {
        init(img, bbox.get_rect());
    }

    void updateTrackerPosition(BBox_c  bbox)
    {
        if (p_resize_image) {
            BBox_c tmp = bbox;
            tmp.scale(0.5);
            p_pose.cx = tmp.cx;
            p_pose.cy = tmp.cy;
        } else {
            p_pose.cx = bbox.cx;
            p_pose.cy = bbox.cy;
        }
    }

    // frame-to-frame object tracking
    void track(Mat img)
    {
        Mat input_gray=new Mat(), input_rgb = img.clone();
        if (img.channels() == 3){
            Imgproc.cvtColor(img, input_gray, Imgproc.COLOR_BGR2GRAY);
            input_gray.convertTo(input_gray, CvType.CV_32FC1);
        }else
            img.convertTo(input_gray, CvType.CV_32FC1);

        // don't need too large image
        if (p_resize_image) {
            Imgproc.resize(input_gray, input_gray, new Size(0, 0), 0.5, 0.5, Imgproc.INTER_AREA);
            Imgproc.resize(input_rgb, input_rgb, new Size(0, 0), 0.5, 0.5, Imgproc.INTER_AREA);
        }

        ArrayList<Mat> patch_feat;
        double max_response = -1.;
        Mat max_response_map=new Mat();
        Point max_response_pt=new Point();
        int scale_index = 0;
        ArrayList<Double> scale_responses=new ArrayList<Double>();
/*
        if (m_use_multithreading){
            std::vector<std::future<cv::Mat>> async_res(p_scales.size());
            for (size_t i = 0; i < p_scales.size(); ++i) {
                async_res[i] = std::async(std::launch::async,
                    [this, &input_gray, &input_rgb, i]() -> cv::Mat
                {
                    std::vector<cv::Mat> patch_feat_async = get_features(input_rgb, input_gray, this->p_pose.cx, this->p_pose.cy, this->p_windows_size[0],
                        this->p_windows_size[1], this->p_current_scale * this->p_scales[i]);
                    ComplexMat zf = fft2(patch_feat_async, this->p_cos_window);
                    if (m_use_linearkernel)
                        return ifft2((p_model_alphaf * zf).sum_over_channels());
                    else {
                        ComplexMat kzf = gaussian_correlation(zf, this->p_model_xf, this->p_kernel_sigma);
                        return ifft2(this->p_model_alphaf * kzf);
                    }
                });
            }

            for (size_t i = 0; i < p_scales.size(); ++i) {
                // wait for result
                async_res[i].wait();
                cv::Mat response = async_res[i].get();

                double min_val, max_val;
                cv::Point2i min_loc, max_loc;
                cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

                double weight = p_scales[i] < 1. ? p_scales[i] : 1./p_scales[i];
                if (max_val*weight > max_response) {
                    max_response = max_val*weight;
                    max_response_map = response;
                    max_response_pt = max_loc;
                    scale_index = i;
                }
                scale_responses.push_back(max_val*weight);
            }
        } else {

        }
 */
        //single thread
        for (int i = 0; i < p_scales.size(); ++i) {
            patch_feat = get_features(input_rgb, input_gray, (int)p_pose.cx, (int)p_pose.cy, p_windows_size[0], p_windows_size[1], p_current_scale * p_scales.get(i));
            ComplexMat zf = fft2(patch_feat, p_cos_window);
            Mat response;
            if (m_use_linearkernel)
                response = ifft2((p_model_alphaf.muln(zf)).sum_over_channels());
            else {
                ComplexMat kzf = gaussian_correlation(zf, p_model_xf, p_kernel_sigma,false);
                response = ifft2(p_model_alphaf.muln(kzf));
            }

            /* target location is at the maximum response. we must take into
            account the fact that, if the target doesn't move, the peak
            will appear at the top-left corner, not at the center (this is
            discussed in the paper). the responses wrap around cyclically. */
            double min_val, max_val;
            Point min_loc, max_loc;
            Mat mask=new Mat();
            Core.MinMaxLocResult re= Core.minMaxLoc(response,mask);
            min_val=re.minVal;
            max_val=re.maxVal;
            min_loc=re.minLoc;
            max_loc=re.maxLoc;

            double weight = p_scales.get(i) < 1. ? p_scales.get(i) : 1./p_scales.get(i);
            if (max_val*weight > max_response) {
                max_response = max_val*weight;
                max_response_map = response;
                max_response_pt = max_loc;
                scale_index = i;
            }
            scale_responses.add(max_val*weight);
        }

        //sub pixel quadratic interpolation from neighbours
        if (max_response_pt.y > max_response_map.rows() / 2) //wrap around to negative half-space of vertical axis
            max_response_pt.y = max_response_pt.y - max_response_map.rows();
        if (max_response_pt.x > max_response_map.cols() / 2) //same for horizontal axis
            max_response_pt.x = max_response_pt.x - max_response_map.cols();

        Point new_location=new Point(max_response_pt.x, max_response_pt.y);

        if (m_use_subpixel_localization)
            new_location = sub_pixel_peak(max_response_pt, max_response_map);

        p_pose.cx += p_current_scale*p_cell_size*new_location.x;
        p_pose.cy += p_current_scale*p_cell_size*new_location.y;
        if (p_pose.cx < 0) p_pose.cx = 0;
        if (p_pose.cx > img.cols()-1) p_pose.cx = img.cols()-1;
        if (p_pose.cy < 0) p_pose.cy = 0;
        if (p_pose.cy > img.rows()-1) p_pose.cy = img.rows()-1;

        //sub grid scale interpolation
        double new_scale = p_scales.get(scale_index);
        if (m_use_subgrid_scale)
            new_scale = sub_grid_scale(scale_responses, scale_index);

        p_current_scale *= new_scale;

        if (p_current_scale < p_min_max_scale[0])
            p_current_scale = p_min_max_scale[0];
        if (p_current_scale > p_min_max_scale[1])
            p_current_scale = p_min_max_scale[1];

        //obtain a subwindow for training at newly estimated target position
        patch_feat = get_features(input_rgb, input_gray, (int)p_pose.cx, (int)p_pose.cy, p_windows_size[0], p_windows_size[1], p_current_scale);
        ComplexMat xf = fft2(patch_feat, p_cos_window);

        //subsequent frames, interpolate model
        p_model_xf = p_model_xf.mulconstant(1. - p_interp_factor).Addcomplexmat( xf.mulconstant(p_interp_factor));

        ComplexMat alphaf_num, alphaf_den;

        if (m_use_linearkernel) {
            ComplexMat xfconj = xf.conj();
            alphaf_num = xfconj.mul(p_yf);
            alphaf_den = (xf.muln(xfconj));
        } else {
            //Kernel Ridge Regression, calculate alphas (in Fourier domain)
            ComplexMat kf = gaussian_correlation(xf, xf, p_kernel_sigma, true);
//        ComplexMat alphaf = p_yf / (kf + p_lambda); //equation for fast training
//        p_model_alphaf = p_model_alphaf * (1. - p_interp_factor) + alphaf * p_interp_factor;
            alphaf_num = p_yf.muln(kf);
            alphaf_den = kf.muln(kf.Addconstant(p_lambda));
        }

        p_model_alphaf_num = p_model_alphaf_num.mulconstant(1. - p_interp_factor).Addcomplexmat(alphaf_num.mulconstant( p_interp_factor));
        p_model_alphaf_den = p_model_alphaf_den.mulconstant(1. - p_interp_factor).Addcomplexmat(alphaf_den.mulconstant(p_interp_factor)) ;
        p_model_alphaf = p_model_alphaf_num.div(p_model_alphaf_den);
    }

    BBox_c getBBox()
    {
        BBox_c tmp = p_pose;
        tmp.w *= p_current_scale;
        tmp.h *= p_current_scale;
        if (p_resize_image)
            tmp.scale(2);
        return tmp;
    }


    //helping functions



    Point sub_pixel_peak(Point max_loc, Mat response)
    {
        //find neighbourhood of max_loc (response is circular)
        // 1 2 3
        // 4   5
        // 6 7 8
        Point p1=new Point(max_loc.x-1, max_loc.y-1), p2=new Point(max_loc.x, max_loc.y-1), p3=new Point(max_loc.x+1, max_loc.y-1);
        Point p4=new Point(max_loc.x-1, max_loc.y), p5=new Point(max_loc.x+1, max_loc.y);
        Point p6=new Point(max_loc.x-1, max_loc.y+1), p7=new Point(max_loc.x, max_loc.y+1), p8=new Point(max_loc.x+1, max_loc.y+1);

        // fit 2d quadratic function f(x, y) = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
        double[] val1={
                p1.x*p1.x, p1.x*p1.y, p1.y*p1.y, p1.x, p1.y, 1.f,
                p2.x*p2.x, p2.x*p2.y, p2.y*p2.y, p2.x, p2.y, 1.f,
                p3.x*p3.x, p3.x*p3.y, p3.y*p3.y, p3.x, p3.y, 1.f,
                p4.x*p4.x, p4.x*p4.y, p4.y*p4.y, p4.x, p4.y, 1.f,
                p5.x*p5.x, p5.x*p5.y, p5.y*p5.y, p5.x, p5.y, 1.f,
                p6.x*p6.x, p6.x*p6.y, p6.y*p6.y, p6.x, p6.y, 1.f,
                p7.x*p7.x, p7.x*p7.y, p7.y*p7.y, p7.x, p7.y, 1.f,
                p8.x*p8.x, p8.x*p8.y, p8.y*p8.y, p8.x, p8.y, 1.f,
                max_loc.x*max_loc.x, max_loc.x*max_loc.y, max_loc.y*max_loc.y, max_loc.x, max_loc.y, 1.f
        };
        double[] val2={
                get_response_circular(p1, response),
                get_response_circular(p2, response),
                get_response_circular(p3, response),
                get_response_circular(p4, response),
                get_response_circular(p5, response),
                get_response_circular(p6, response),
                get_response_circular(p7, response),
                get_response_circular(p8, response),
                get_response_circular(max_loc, response)
        };
        Mat A = new Mat(9,6, CvType.CV_32FC1);
        A.put(0,0,val1);
        Mat fval = new Mat(9,1, CvType.CV_32FC1);
        fval.put(0,0,val2);
        Mat x=new Mat();
        Core.solve(A, fval, x, Core.DECOMP_SVD);

        double[] r=new double[5];
        x.get(0,0,r);
        double a=r[0],b=r[1],c=r[2],d=r[3],e=r[4];
        Point sub_peak=new Point(max_loc.x, max_loc.y);
        if (b > 0 || b < 0) {
            sub_peak.y = ((2.f * a * e) / b - d) / (b - (4 * a * c) / b);
            sub_peak.x = (-2 * c * sub_peak.y - e) / b;
        }

        return sub_peak;
    }

    double sub_grid_scale(ArrayList<Double> responses, int index)
    {
        index=-1;
        Mat A=new Mat();
        Mat fval=new Mat();
        if (index < 0 || index > (int)p_scales.size()-1) {
            // interpolate from all values
            // fit 1d quadratic function f(x) = a*x^2 + b*x + c
            A.create(p_scales.size(), 3, CvType.CV_32FC1);
            fval.create(p_scales.size(), 1, CvType.CV_32FC1);
            for (int i = 0; i < p_scales.size(); ++i) {
                A.put(i, 0,p_scales.get(i) * p_scales.get(i));
                A.put(i, 1,p_scales.get(i));
                A.put(i, 2,1);
                fval.put(i,0, responses.get(i));
            }
        } else {
            //only from neighbours
            if (index == 0 || index == (int)p_scales.size()-1)
                return p_scales.get(index);
            A.create(p_scales.size(), 3, CvType.CV_32FC1);
            fval.create(p_scales.size(), 1, CvType.CV_32FC1);
            double[] f1={
                    p_scales.get(index-1) * p_scales.get(index-1), p_scales.get(index-1), 1,
                    p_scales.get(index) * p_scales.get(index), p_scales.get(index), 1,
                    p_scales.get(index+1) * p_scales.get(index+1), p_scales.get(index+1), 1
            };
            A.put(0,0,f1);
            double[] f2={
                    responses.get(index-1), responses.get(index), responses.get(index+1)
            };
            fval.put(0,0,f2);
        }

        Mat x=new Mat();
        Core.solve(A, fval, x, Core.DECOMP_SVD);
        double[] t=new double[2];
        x.get(0,0,t);
        double a = t[0], b = t[1];
        double scale = p_scales.get(index);
        if (a > 0 || a < 0)
            scale = -b / (2 * a);
        return scale;
    }

    float get_response_circular(Point pt, Mat response)
    {
        int x =  (int)pt.x;
        int y =  (int)pt.y;
        if (x < 0)
            x = response.cols() + x;
        if (y < 0)
            y = response.rows() + y;
        if (x >= response.cols())
            x = x - response.cols();
        if (y >= response.rows())
            y = y - response.rows();

        float[] val=new float[1];
        return response.get(y,x,val);
    }

}
