package com.example.objecttrack;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.annotation.NonNull;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class VideoitemAdapter extends ArrayAdapter{
    private final int resourceId;
    Context s;
    public VideoitemAdapter(Context context, int textViewResourceId, List<Videoitem> objects) {
        super(context, textViewResourceId, objects);
        resourceId = textViewResourceId;
    }
    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        Videoitem vi = (Videoitem) getItem(position); // 获取当前项的Fruit实例
        View view = LayoutInflater.from(getContext()).inflate(resourceId, null);//实例化一个对象
        ImageView fruitImage =  view.findViewById(R.id.video_image);//获取该布局内的图片视图
        TextView fruitName =  view.findViewById(R.id.video_name);//获取该布局内的文本视图
        fruitImage.setImageBitmap(vi.getImage());//为图片视图设置图片资源
        fruitName.setText(vi.getName());//为文本视图设置文本内容
        return view;

    }
}
