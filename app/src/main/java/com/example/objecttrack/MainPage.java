package com.example.objecttrack;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.Toolbar;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.Adapter;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.ListView;
import android.widget.Toast;

import org.opencv.video.Video;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class MainPage extends AppCompatActivity {

    final MainPage m=this;
    private ArrayList<Videoitem> l=new ArrayList<Videoitem>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        supportRequestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.main_page);
        Toolbar toolbar = findViewById(R.id.toolbar);
        toolbar.setTitle("Video List");
        setSupportActionBar(toolbar);
        toolbar.setOnMenuItemClickListener(new Toolbar.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                switch (item.getItemId()) {
                    case R.id.cameravideo:
                        Intent intent=new Intent(MainPage.this,MainActivity.class);
                        startActivity(intent);
                        break;
                    case R.id.downloadvideo:
                        Toast.makeText(MainPage.this, "download !", Toast.LENGTH_SHORT).show();
                        break;
                }
                return true;
            }
        });
        try {
            initialvideolist();
            VideoitemAdapter adapter = new VideoitemAdapter(MainPage.this, R.layout.videoitem,l );
            ListView listView = findViewById(R.id.listview);
            listView.setAdapter(adapter);
            listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                @Override
                public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                    Intent intent=new Intent(m, VideoActivity.class);
                    Videoitem v=(Videoitem)adapterView.getItemAtPosition(i);
                    String name=v.getName();
                    intent.putExtra("name", name);
                    startActivity(intent);
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    void initialvideolist() throws IOException {
        String[] files=getResources().getAssets().list("images");
        for(int i=0;i<files.length;i++)
        {
            if(!files[i].endsWith("_dir"))
                continue;
            String name=files[i].substring(0,files[i].length()-4);
            String path="images/"+files[i]+"/img/0001.jpg";
            InputStream in =getAssets().open(path);
            Bitmap bm= BitmapFactory.decodeStream(in);
            Videoitem a=new Videoitem(name,bm);
            l.add(a);
        }
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_menu, menu);
        return true;
    }
}
