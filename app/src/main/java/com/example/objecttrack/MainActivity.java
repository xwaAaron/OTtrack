package com.example.objecttrack;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener, CameraBridgeViewBase.CvCameraViewListener2{

    JavaCameraView objectTrackingView;
    private static final Scalar TRACKING_RECT_COLOR = new Scalar(255, 255, 0, 255);
    ObjectTracker objectTracker;  //  目标追踪器
    Rect mTrackWindow;  // 追踪目标区域
    boolean isTracking;  // 追踪状态
    double mCameraArea;
    Mat mRgba;
    Mat mGray;
    int xDown;
    int yDown;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case BaseLoaderCallback.SUCCESS:
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        if (null == objectTracker) {
            objectTracker = new ObjectTracker() ;
        }
        if (null != mTrackWindow) {
            RotatedRect rotatedRect = objectTracker.objectTracking(mRgba);
            Imgproc.ellipse(mRgba, rotatedRect, TRACKING_RECT_COLOR, 6);
            Rect rect = rotatedRect.boundingRect();
            Imgproc.rectangle(mRgba, rect.tl(), rect.br(), TRACKING_RECT_COLOR, 3);
        }
        return mRgba;
    }


    public boolean onTouch(View v, MotionEvent event) {
        if (null == mRgba) {
            return true;
        }
        int cols = mRgba.cols();
        int rows = mRgba.rows();
        int w=objectTrackingView.getWidth();
        int h=objectTrackingView.getHeight();
        int xOffset = (w - cols) / 2;
        int yOffset = (h - rows) / 2;
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                isTracking = false;
                xDown = (int) event.getX() - xOffset;
                yDown = (int) event.getY() - yOffset;
                break;
            case MotionEvent.ACTION_UP:
                int xUp = (int) event.getX() - xOffset;
                int yUp = (int) event.getY() - yOffset;
                int width = Math.abs(xUp - xDown);
                int height = Math.abs(yUp - yDown);
                if (0 == width || 0 == height) {
                    Toast.makeText(MainActivity.this, "Goal Too Small !", Toast.LENGTH_SHORT).show();
                    break;
                }
                // 获取跟踪目标
                mTrackWindow = new Rect(Math.min(xDown, xUp), Math.min(yDown, yUp), width, height);
                // 创建跟踪目标
                objectTracker = new ObjectTracker();
                objectTracker.createTrackedObject(mRgba, mTrackWindow);
                isTracking = true;
                Toast.makeText(MainActivity.this, "Tracking Goal Selected !", Toast.LENGTH_SHORT).show();
                break;
            default:
                break;
        }
        return true;
    }

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        objectTrackingView = (JavaCameraView) findViewById(R.id.tracking_view);
        objectTrackingView.setCvCameraViewListener(this);
        objectTrackingView.setOnTouchListener(MainActivity.this);
        objectTrackingView.enableView();
    }

    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
}


