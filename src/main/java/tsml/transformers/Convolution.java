package tsml.transformers;

import java.util.Arrays;

import tsml.data_containers.TimeSeriesInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Convolution implements Transformer {

    public enum ConvolutionType {
        FILL, SYMM
    };

    double[] kernel1D;
    double[][] kernel2D;

    ConvolutionType convType = ConvolutionType.FILL;
    double padValue = 0;

    public Convolution(double[][] kernel) {
        kernel2D = kernel;
    }

    public Convolution(double[] kernel) {
        kernel1D = kernel;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Instance transform(Instance inst) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // TODO Auto-generated method stub
        return null;
    }

    public double[] convolution1D(double[] data) {
        return convolution1D(data, this.kernel1D);
    }

    public static double[] convolution1D(double[] data, double[] kernel1D) {

        double[] out = new double[kernel1D.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = 0; // set to zero before sum
            for (int j = 0; j < kernel1D.length; j++) {
                out[i] += data[i - j] * kernel1D[j]; // convolve: multiply and accumulate
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
    }

}
