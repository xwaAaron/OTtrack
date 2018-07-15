package com.example.objecttrack;

import android.graphics.Bitmap;

public class Videoitem {
    private String name;
    private Bitmap image;

    public Videoitem(String name, Bitmap image) {
        this.name = name;
        this.image = image;
    }

    public String getName() {
        return name;
    }

    public Bitmap getImage() {
        return image;
    }
}
