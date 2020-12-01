package tsml.transformers;

import java.util.Arrays;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.utilities.Converter;
import weka.core.Instance;
import weka.core.Instances;

public class Convolution implements Transformer {

    public enum ConvolutionType {
        FILL, SYMM
    };

    double[] kernel1D = null;
    double[][] kernel2D = null;

    ConvolutionType convType = ConvolutionType.FILL;
    double padValue = 0;

    public Convolution(int kernelSize1D, double constantValue){
        kernel1D = new double[kernelSize1D];
        Arrays.fill(kernel1D, constantValue);
    }

    public Convolution(int kernelSize2DX, int kernelSize2DY, double constantValue){
        kernel2D = new double[kernelSize2DX][kernelSize2DY];
        for(int i=0; i<kernel2D.length; i++)
            Arrays.fill(kernel2D[i], constantValue);
    }


    public Convolution(double[][] kernel) {
        kernel2D = kernel;
    }

    public Convolution(double[] kernel) {
        kernel1D = kernel;
    }

    public void setPad(double pad){
        convType = ConvolutionType.FILL;
        padValue = pad;
    }

    public void setSymm(){
        convType = ConvolutionType.SYMM;
    }


    //TODO: WEKA Version. Bleh
    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Instance transform(Instance inst) {
        return Converter.toArff(transform(Converter.fromArff(inst)));
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out;
        if(kernel1D != null){
            out = new double[inst.getNumDimensions()][];
            int i = 0;
            for (TimeSeries ts : inst) {
                out[i++] = convolution1D(ts.toValueArray());
            }
        }
        else{
            out = convolution2D(inst.toValueArray());
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    public double[] convolution1D(double[] data) {
        return convolution1D(data, this.kernel1D, convType, padValue);
    }

    public static double[] convolution1D(double[] data, double[] kernel1D, ConvolutionType convType, double padValue) {
        int kCenter = kernel1D.length / 2;
        double[] out = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < kernel1D.length; j++) {
                int mm = kernel1D.length - 1 - j; // row index of flipped kernel
                int ii = i + (kCenter - mm);

                boolean iPad = ii >= 0 && ii < data.length;
                // ignore input samples which are out of bound
                if (iPad)
                    out[i] += data[ii] * kernel1D[mm];
                else {
                    if (convType == ConvolutionType.SYMM) {

                        // symmetrical flip the values.
                        if (!iPad)
                            ii = i - (kCenter - mm);

                        out[i] += data[ii] * kernel1D[mm];
                    }
                    // else always just do a pad
                    else {
                        out[i] += padValue * kernel1D[mm];
                    }
                }

            }
        }

        return out;
    }

    public double[][] convolution2D(double[][] data) {
        return convolution2D(data, this.kernel2D, convType, padValue);
    }

    public static double[][] convolution2D(double[][] data, double[][] kernel2D, ConvolutionType convType,
            double padValue) {

        double[][] out = new double[data.length][];

        // find center position of kernel (half of kernel size)
        int kCenterX = kernel2D.length / 2;
        int kCenterY = kernel2D[0].length / 2;

        for (int i = 0; i < data.length; ++i) {
            out[i] = new double[data[i].length];
            for (int j = 0; j < data[i].length; ++j) {
                for (int m = 0; m < kernel2D.length; ++m) {
                    int mm = kernel2D.length - 1 - m; // row index of flipped kernel
                    for (int n = 0; n < kernel2D[mm].length; ++n) {
                        int nn = kernel2D[mm].length - 1 - n; // column index of flipped kernel

                        int ii = i + (kCenterY - mm);
                        int jj = j + (kCenterX - nn);

                        boolean iPad = ii >= 0 && ii < data.length;
                        boolean jPad = jj >= 0 && jj < data[i].length;
                        // ignore input samples which are out of bound
                        if (iPad && jPad)
                            out[i][j] += data[ii][jj] * kernel2D[mm][nn];
                        else {
                            if (convType == ConvolutionType.SYMM) {

                                // symmetrical flip the values.
                                if (!iPad)
                                    ii = i - (kCenterY - mm);

                                if (!jPad)
                                    jj = j - (kCenterX - nn);
                                    
                                out[i][j] += data[ii][jj] * kernel2D[mm][nn];
                            }
                            // else always just do a pad
                            else {
                                out[i][j] += padValue * kernel2D[mm][nn];
                            }
                        }
                    }
                }
            }
        }

        return out;
    }

    public static void main(String[] args) {
        // 3x3 averaging kernel
        double[][] kernel = { { 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0 }, { 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0 },
                { 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0 } };

        // 10x10 data.
        double[][] data = { { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 } };

        double[][] out = convolution2D(data, kernel, ConvolutionType.FILL, 1);
        System.out.println(Arrays.deepToString(out));


        TimeSeriesInstance ts = new TimeSeriesInstance(data);
        Convolution conv = new Convolution(3, 3, 1.0/9.0);
        conv.setPad(1);
        TimeSeriesInstance out_ts = conv.transform(ts);
        System.out.println(out_ts);

        double[][] data1 = { { 0.0, 1.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0 } };
        ts = new TimeSeriesInstance(data1);
        conv = new Convolution(3, 1.0/3.0);
        //conv.setPad(1);
        conv.setSymm();
        out_ts = conv.transform(ts);
        System.out.println(out_ts);



    }

}
