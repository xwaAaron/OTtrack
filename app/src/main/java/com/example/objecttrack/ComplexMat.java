package com.example.objecttrack;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Iterator;


class ComplexMat
{
    public int cols;
    public int rows;
    public int n_channels;
    private ArrayList<ArrayList<Complex>> p_data=new ArrayList<ArrayList<Complex>>();


    class Complex
    {
        double real;
        double imag;
        Complex(){
            real=imag=0;
        }
        Complex(float r,float i)
        {
            real=r;
            imag=i;
        }
        double real()
        {
            return real;
        }
        double imag()
        {
            return imag;
        }
        Complex div(Complex c2)
        {
            Complex c=this;
            c.real=(real*c2.real+imag*c2.imag)/(c2.real*c2.real+c2.imag*c2.imag);
            c.imag=(imag*c2.real-real*c2.imag)/(c2.real*c2.real+c2.imag*c2.imag);
            return c;
        }
    }

    ComplexMat()
    {
        cols=0;
        rows=0;
        n_channels=0;
    }

    ComplexMat(int _rows, int _cols, int _n_channels)
    {
        cols=_cols;
        rows=_rows;
        n_channels=_n_channels;
    }

    //assuming that mat has 2 channels (real, img)
    ComplexMat(Mat mat)
    {
        cols=mat.cols();
        rows=mat.rows();
        n_channels=1;
        p_data.add(convert(mat));
    }

    ComplexMat(ComplexMat x)
    {
        cols=x.cols;
        rows=x.rows;
        n_channels=x.n_channels;
        this.p_data=new ArrayList<ArrayList<Complex>>();
        for(int i=0;i<n_channels;i++)
        {
            ArrayList<Complex> list=new ArrayList<Complex>();
            for(int j=0;j<cols*rows;j++)
            {
                Complex a=new Complex((float)this.p_data.get(i).get(j).real,(float)this.p_data.get(i).get(j).imag);
                list.add(a);
            }
            this.p_data.add(list);
        }
    }

    //assuming that mat has 2 channels (real, imag)
    void set_channel(int idx, Mat mat)
    {
        assert(idx >= 0 && idx < n_channels);
        p_data.set(idx,convert(mat));
    }

    //compute the quatratic sum of all the real part and image part
    float sqr_norm()
    {
        float sum_sqr_norm = 0;
        for (int i = 0; i < n_channels; ++i)
        {
            ArrayList<Complex> temp=p_data.get(i);
            Iterator lhs=temp.iterator();
            while(lhs.hasNext())
            {
                Complex tt=(Complex)lhs.next();
                sum_sqr_norm += tt.real()*tt.real() + tt.imag()*tt.imag();
            }
        }
        //std::for_each(p_data[i].begin(), p_data[i].end(), [&sum_sqr_norm](const std::complex<T> & c) { sum_sqr_norm += c.real()*c.real() + c.imag()*c.imag(); } );
        return sum_sqr_norm / (float)(cols*rows);
    }

    //assign the quatratic sum of the real part and image part to the real part
    ComplexMat sqr_mag()
    {
        ComplexMat result = this;
        for (int i = 0; i < n_channels; ++i)
        {
            ArrayList<Complex> temp=result.p_data.get(i);
            Iterator lhs=temp.iterator();
            while(lhs.hasNext())
            {
                Complex tt=(Complex)lhs.next();
                tt.real = tt.real()*tt.real() + tt.imag()*tt.imag();
                tt.imag=0;
            }
        }
        return result;
    }

    //reverse the image part
    ComplexMat conj()
    {
        ComplexMat result = this;
        for (int i = 0; i < n_channels; ++i)
        {
            ArrayList<Complex> temp=result.p_data.get(i);
            Iterator lhs=temp.iterator();
            while(lhs.hasNext())
            {
                Complex tt=(Complex)lhs.next();
                tt.real = tt.real();
                tt.imag = -1*tt.imag();
            }
        }
        return result;
    }

    // add values of all the channels to the first channel
    ComplexMat sum_over_channels()
    {
        assert(p_data.size() > 1);
        ComplexMat result=new ComplexMat(this.rows, this.cols, 1);
        result.p_data.add(new ArrayList<Complex>(this.p_data.get(0))) ;
        for (int i = 1; i < n_channels; ++i)
        {
            int count=result.p_data.get(0).size();
            for(int j=0;j<count;j++)
            {
                result.p_data.get(0).get(j).real+=this.p_data.get(i).get(j).real;
                result.p_data.get(0).get(j).imag+=this.p_data.get(i).get(j).imag;
            }
        }
        return result;
    }

    //return 2 channels (real, imag) for first complex channel
    Mat to_cv_mat()
    {
        assert(p_data.size() >= 1);
        return channel_to_cv_mat(0);
    }

    //return a vector of 2 channels (real, imag) per one complex channel
    ArrayList<Mat> to_cv_mat_vector()
    {
        ArrayList<Mat> result=new ArrayList<Mat>();
        for (int i = 0; i < n_channels; ++i)
            result.add(channel_to_cv_mat(i));
        return result;
    }
/*
    //element-wise per channel multiplication, division and addition

    ComplexMat_<T> operator/(const ComplexMat_<T> & rhs) const
    {
        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs) { c_lhs /= c_rhs; }, rhs);
    }
    ComplexMat_<T> operator+(const ComplexMat_<T> & rhs) const
    {
        return mat_mat_operator( [](std::complex<T> & c_lhs, const std::complex<T> & c_rhs)  { c_lhs += c_rhs; }, rhs);
    }

    //multiplying or adding constant
    ComplexMat_<T> operator*(const T & rhs) const
    {
        return mat_const_operator( [&rhs](std::complex<T> & c) { c *= rhs; });
    }
    ComplexMat_<T> operator+(const T & rhs) const
    {
        return mat_const_operator( [&rhs](std::complex<T> & c) { c += rhs; });
    }
*/
    //multiplying element-wise multichannel by one channel mats (rhs mat is with one channel)

    // matrix mul matrix  overload operator *

    ComplexMat div(ComplexMat rhs)
    {
        assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

        ComplexMat result =new ComplexMat(this);
        for (int i = 0; i < n_channels; ++i) {
            for (int j = 0; j < cols * rows; j++) {
                Complex a=result.p_data.get(i).get(j);
                Complex b=rhs.p_data.get(i).get(j);
                a=a.div(b);
            }
        }
        return result;
    }


    ComplexMat Addconstant(double x)
    {
        ComplexMat res=new ComplexMat(this);
        for (int i = 0; i < n_channels; ++i) {
            for(int j=0;j<this.p_data.size();j++)
            {
                res.p_data.get(i).get(j).real+=x;
            }
        }
        return res;
    }

    ComplexMat Addcomplexmat(ComplexMat x)
    {
        ComplexMat res=new ComplexMat(this);
        for(int i=0;i<n_channels;i++)
        {
            for(int j=0;j<cols*rows;j++)
            {
                Complex a=res.p_data.get(i).get(j);
                Complex b=x.p_data.get(i).get(j);
                a.real+=b.real;
                a.imag+=b.imag;
            }
        }
        return res;
    }

    ComplexMat muln(ComplexMat  rhs)
    {
        assert(rhs.n_channels == n_channels && rhs.cols == cols && rhs.rows == rows);

        ComplexMat result = this;
        for (int i = 0; i < n_channels; ++i) {
            Iterator<Complex> it1=result.p_data.get(i).iterator();
            Iterator<Complex> it2=rhs.p_data.get(0).iterator();
            while(it1.hasNext())
            {
                Complex a=it1.next();
                Complex b=it2.next();
                double newreal=a.real*b.real-a.imag*b.imag;
                double newimag=a.imag*b.real+a.real*b.imag;
                a.real=newreal;
                a.imag=newimag;
            }
        }
        return result;
    }
    // matrix mul vector
    ComplexMat mul(ComplexMat rhs)
    {
        assert(rhs.n_channels == 1 && rhs.cols == cols && rhs.rows == rows);

        ComplexMat result = this;
        for (int i = 0; i < n_channels; ++i) {
            Iterator<Complex> it1=result.p_data.get(i).iterator();
            Iterator<Complex> it2=rhs.p_data.get(0).iterator();
            while(it1.hasNext())
            {
                Complex a=it1.next();
                Complex b=it2.next();
                double newreal=a.real*b.real-a.imag*b.imag;
                double newimag=a.imag*b.real+a.real*b.imag;
                a.real=newreal;
                a.imag=newimag;
            }
        }
       return result;
    }

/*
    //text output
    friend std::ostream & operator<<(std::ostream & os, const ComplexMat_<T> & mat)
    {
        //for (int i = 0; i < mat.n_channels; ++i){
        for (int i = 0; i < 1; ++i){
            os << "Channel " << i << std::endl;
            for (int j = 0; j < mat.rows; ++j) {
                for (int k = 0; k < mat.cols-1; ++k)
                    os << mat.p_data[i][j*mat.cols + k] << ", ";
                os << mat.p_data[i][j*mat.cols + mat.cols-1] << std::endl;
            }
        }
        return os;
    }
*/

    //convert 2 channel mat (real, imag) to vector row-by-row
    ArrayList<Complex> convert(Mat mat)
    {
        ArrayList<Complex> result=new ArrayList<Complex>();
        //result.reserve(mat.cols*mat.rows);
        for (int y = 0; y < mat.rows(); ++y) {
            for (int x = 0; x < mat.cols(); x++){
                float[] f=new float[2];
                mat.get(y,x,f);
                result.add(new Complex(f[0],f[1]));
            }
        }
        return result;
    }

/*    ComplexMat mat_mat_operator(void (*op)(Complex c_lhs, Complex c_rhs), ComplexMat mat_rhs)
    {
        assert(mat_rhs.n_channels == n_channels && mat_rhs.cols == cols && mat_rhs.rows == rows);

        ComplexMat result = *this;
        for (int i = 0; i < n_channels; ++i) {
            auto lhs = result.p_data[i].begin();
            auto rhs = mat_rhs.p_data[i].begin();
            for ( ; lhs != result.p_data[i].end(); ++lhs, ++rhs)
                op(*lhs, *rhs);
        }

        return result;
    }

    ComplexMat matn_mat1_operator(void (*op)(std::complex<T> & c_lhs, const std::complex<T> & c_rhs), const ComplexMat_<T> & mat_rhs)
    {
        assert(mat_rhs.n_channels == 1 && mat_rhs.cols == cols && mat_rhs.rows == rows);

        ComplexMat_<T> result = *this;
        for (int i = 0; i < n_channels; ++i) {
            auto lhs = result.p_data[i].begin();
            auto rhs = mat_rhs.p_data[0].begin();
            for ( ; lhs != result.p_data[i].end(); ++lhs, ++rhs)
                op(*lhs, *rhs);
        }

        return result;
    }

    ComplexMat mat_const_operator(const std::function<void(std::complex<T> & c_rhs)> & op)
    {
        ComplexMat result = *this;
        for (int i = 0; i < n_channels; ++i)
            for (auto lhs = result.p_data[i].begin(); lhs != result.p_data[i].end(); ++lhs)
                op(*lhs);
        return result;
    }
*/
    Mat channel_to_cv_mat(int channel_id)
    {
        Mat result=new Mat(rows, cols, CvType.CV_32FC2);
        int data_id = 0;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; x++){
                double[] f=new double[2];
                f[0]=this.p_data.get(channel_id).get(data_id).real;
                f[1]=this.p_data.get(channel_id).get(data_id++).imag;
                result.put(y,x,f);
            }
        }
        return result;
    }

    ComplexMat mulconstant(double x)
    {
        ComplexMat res=new ComplexMat(this);
        for(int k=0;k<n_channels;k++)
        {
            for(int i=0;i<res.cols*res.rows;i++)
            {
                Complex c=res.p_data.get(k).get(i);
                double real=c.real();
                double imag=c.imag();
                c.real=real*x;
                c.imag=imag*x;
            }
        }
        return res;
    }

};


