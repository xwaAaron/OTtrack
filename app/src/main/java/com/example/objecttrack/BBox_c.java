package com.example.objecttrack;

import org.opencv.core.Rect;

class BBox_c{
        double   cx, cy,w, h;

        void scale(double factor)
        {
            cx *= factor;
            cy *= factor;
            w  *= factor;
            h  *= factor;
        }

        Rect get_rect()
        {
            int x=new Double(cx-w/2).intValue();
            int y=new Double(cy-h/2).intValue();
            int w=new Double(this.w).intValue();
            int h=new Double(this.h).intValue();
            return new Rect(x, y, w, h);
        }
};
