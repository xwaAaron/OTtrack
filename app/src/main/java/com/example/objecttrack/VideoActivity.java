package com.example.objecttrack;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;

public class VideoActivity extends AppCompatActivity {

    private ImageView iv;
    private Handler handler=new Handler();
    private Runnable runnable;
    BufferedReader buffreader;
    String path;
    private int frameint=1;
    private int count=1;
    private String TAG="ottrack";
    private static final Scalar TRACKING_RECT_COLOR = new Scalar(255, 255, 0, 255);
    Mat mRgba;
    Mat mGray;
    boolean isTracking;
    org.opencv.core.Rect mTrackWindow;  // 追踪目标区域
    int xDown;
    int yDown;
    ObjectTracker objectTracker;  //  目标追踪器
    RotatedRect rotatedRect;
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
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video);
        path= getIntent().getStringExtra("name");
        path="images/"+path+"_dir/img";
        iv=findViewById(R.id.imageView);
        iv.setOnTouchListener(new Mytouchlistener());
        try {
            String[] files=getAssets().list(path);
            count=files.length;
        } catch (IOException e) {
            e.printStackTrace();
        }
        startdraw();

    }

    void startdraw()
    {
        runnable=new Runnable(){
            public void run() {
                try {
                    handler.postDelayed(this, 100);
                    DecimalFormat d=new DecimalFormat("0000");
                    String file=path+"/"+d.format(frameint)+".jpg";
                    InputStream in=getAssets().open(file);
                    Bitmap bm= BitmapFactory.decodeStream(in);
                    int w=960,h=720;
                    Bitmap board = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                    Canvas canvas=new Canvas(board);
                    canvas.drawBitmap(bm,(w-bm.getWidth())/2,(h-bm.getHeight())/2,null);
                    Bitmap res=drawtarget(board);
                    iv.setImageBitmap(res);
                    frameint++;
                    Log.i(TAG, "a");
                    if(frameint==count)
                        handler.removeCallbacks(runnable);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        handler.postDelayed(runnable, 0);
    }

    Bitmap drawtarget(Bitmap board){
        mRgba=new Mat(board.getHeight(),board.getWidth(),CvType.CV_8UC4);
        Utils.bitmapToMat(board,mRgba);
        if (null == objectTracker) {
            objectTracker = new ObjectTracker() ;
        }
        if (null != mTrackWindow) {
            RotatedRect rotatedRect = objectTracker.objectTracking(mRgba);
            Imgproc.ellipse(mRgba, rotatedRect, TRACKING_RECT_COLOR, 6);
            org.opencv.core.Rect rect = rotatedRect.boundingRect();
            Imgproc.rectangle(mRgba, rect.tl(), rect.br(), TRACKING_RECT_COLOR, 3);
        }
        Bitmap bitmap = Bitmap.createBitmap(mRgba.width(), mRgba.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mRgba, bitmap);
        return  bitmap;
    }


    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    class Mytouchlistener implements View.OnTouchListener{

        @Override
        public boolean onTouch(View view, MotionEvent event) {
            if (null == mRgba) {
                return true;
            }
            int cols = mRgba.cols();
            int rows = mRgba.rows();
            int w=iv.getWidth();
            int h=iv.getHeight();
            int xOffset = (iv.getWidth() - cols) / 2;
            int yOffset = (iv.getHeight() - rows) / 2;
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
                        Toast.makeText(VideoActivity.this, "Goal Too Small !", Toast.LENGTH_SHORT).show();
                        break;
                    }
                    // 获取跟踪目标
                    mTrackWindow = new org.opencv.core.Rect(Math.min(xDown, xUp), Math.min(yDown, yUp), width, height);
                    // 创建跟踪目标
                    objectTracker = new ObjectTracker();
                    objectTracker.createTrackedObject(mRgba, mTrackWindow);
                    isTracking = true;
                    Toast.makeText(VideoActivity.this, "Tracking Goal Selected !", Toast.LENGTH_SHORT).show();
                    break;
                default:
                    break;
            }
            return true;
        }
    }
}