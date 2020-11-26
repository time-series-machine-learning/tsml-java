package tsml.transformers;

import experiments.data.DatasetLists;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import utilities.InstanceTools;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

import static experiments.data.DatasetLoading.loadDataNullable;
import static org.apache.commons.math3.transform.TransformType.FORWARD;

public class AudioFeatures implements Transformer {

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 1; i < 7; i++) {
            atts.add(new Attribute("AF_att" + i));
        }
        atts.add(inputFormat.classAttribute());
        Instances transformHeader = new Instances("audioTransform", atts, inputFormat.numInstances());
        transformHeader.setClassIndex(transformHeader.numAttributes() - 1);
        return transformHeader;
    }

    @Override
    public Instance transform(Instance inst) {
        return new DenseInstance(1,
                audioTransform(InstanceTools.ConvertInstanceToArrayRemovingClassValue(inst), inst.classValue()));
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // could do this across all dimensions.
        double[][] out = new double[inst.getNumDimensions()][];
        int i = 0;
        for (TimeSeries ts : inst) {
            out[i++] = audioTransform(ts.toValueArray());
        }

        // create a new output instance with the ACF data.
        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    private double[] audioTransform(double[] series, double classVal) {
        return audioTransform(series, classVal, true);
    }

    private double[] audioTransform(double[] series) {
        return audioTransform(series, Double.NaN, false);
    }

    private double[] audioTransform(double[] series, double classVal, boolean includeClassVal) {
        int fs = (series.length + 1) * 2;

        int nfft = fs;
        nfft = nearestPowerOF2(nfft);
        Complex[] complexData = new Complex[nfft];
        double[] spectralMag = new double[nfft / 2];
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);

        for (int i = 0; i < complexData.length; i++) {
            complexData[i] = new Complex(0.0, 0.0);
        }

        double mean = 0;
        if (series.length < nfft) {
            for (int i = 0; i < series.length; i++) {
                mean += series[i];
            }
            mean /= series.length;
        }

        for (int i = 0; i < nfft; i++) {
            if (i < series.length)
                complexData[i] = new Complex(series[i], 0);
            else
                complexData[i] = new Complex(mean, 0);
        }

        complexData = fft.transform(complexData, FORWARD);

        for (int i = 0; i < (nfft / 2); i++) {
            spectralMag[i] = complexData[i].abs();
        }



        double[] output = new double[includeClassVal ? 7 : 6];
        output[0] = spectralCentroid(spectralMag, nfft);
        output[1] = spectralSpread(spectralMag, nfft);
        output[2] = spectralFlatness(spectralMag);
        output[3] = spectralSkewness(spectralMag);
        output[4] = spectralKurtosis(spectralMag);
        output[5] = zeroCrossingRate(series, nfft);
        if(includeClassVal)
            output[6] = classVal;
        return output;
    }

    private static int nearestPowerOF2(int x) {
        float power = (float) (Math.log(x) / Math.log(2));
        int m = (int) Math.ceil(power);
        return (int) Math.pow(2.0, (double) m);
    }

    private static double spectralCentroid(double[] spectralMag, int fs) {
        int nfft = fs;
        double numerator = 0.0;
        double denominator = 0.0;
        double binWidth = ((double) fs / (double) nfft);
        for (int i = 0; i < spectralMag.length; i++) {
            numerator += (((i * binWidth) + (binWidth / 2)) * Math.pow(spectralMag[i], 2));
            denominator += Math.pow(spectralMag[i], 2);
        }

        return numerator / denominator;
    }

    private static double spectralSpread(double[] spectralMag, int fs) {
        int nfft = fs;
        double numerator = 0.0;
        double denominator = 0.0;
        double binWidth = ((double) fs / (double) nfft);
        double spectralCentroid = spectralCentroid(spectralMag, fs);
        for (int i = 0; i < spectralMag.length; i++) {
            numerator += (Math.pow(((i * binWidth) + (binWidth / 2)) - spectralCentroid, 2)
                    * Math.pow(spectralMag[i], 2));
            denominator += Math.pow(spectralMag[i], 2);
        }

        return Math.sqrt(numerator / denominator);
    }

    private static double spectralFlatness(double[] spectralMag) {
        int numBands = (10 > spectralMag.length ? spectralMag.length : 10);
        double numerator = 0.0;
        double denominator = 0.0;
        double spectralFlatness = 0.0;
        int bandWidth = spectralMag.length / numBands;

        for (int i = 0; i < numBands; i++) {
            double numeratorTmp = 0.0;
            double denominatorTmp = 0.0;
            for (int j = (i * bandWidth); j < (i * bandWidth) + bandWidth; j++) {
                numeratorTmp *= Math.pow(spectralMag[j], 2);
                denominatorTmp += Math.pow(spectralMag[j], 2);
            }
            numerator = Math.pow(numeratorTmp, (1 / bandWidth));
            denominator = ((double) 1 / (double) bandWidth) * denominatorTmp;
            spectralFlatness += (numerator / denominator);
        }

        return spectralFlatness / numBands;
    }

    private static double spectralSkewness(double[] spectralMag) {
        double numerator = 0.0;
        double denominator = 0.0;
        double spectralMean = 0.0;

        for (int i = 0; i < spectralMag.length; i++) {
            spectralMean += spectralMag[i];
        }
        spectralMean /= spectralMag.length;

        for (int i = 0; i < spectralMag.length; i++) {
            numerator += Math.pow((spectralMag[i] - spectralMean), 3);
            denominator += Math.pow((spectralMag[i] - spectralMean), 2);
        }
        numerator = ((double) 1 / (double) spectralMag.length) * numerator;
        denominator = Math.pow(((double) 1 / (double) spectralMag.length) * denominator, (3 / 2));
        return numerator / denominator;
    }

    private static double spectralKurtosis(double[] spectralMag) {
        double numerator = 0.0;
        double denominator = 0.0;
        double spectralMean = 0.0;
        double fourthMoment = 0.0;

        for (int i = 0; i < spectralMag.length; i++) {
            spectralMean += spectralMag[i];
        }
        spectralMean /= spectralMag.length;

        for (int i = 0; i < spectralMag.length; i++) {
            fourthMoment += Math.pow(spectralMag[i] - spectralMean, 4);
        }
        fourthMoment /= (spectralMag.length - 1);

        for (int i = 0; i < spectralMag.length; i++) {
            numerator += Math.pow((spectralMag[i] - spectralMean), 4);
        }
        denominator = fourthMoment * (spectralMag.length - 1);

        return (numerator / denominator) - 3;
    }

    private static double zeroCrossingRate(double[] series, int fs) {
        double zcr = 0.0;
        for (int i = 1; i < series.length; i++) {
            zcr += Math.abs((series[i] >= 0 ? 1 : 0) - (series[i - 1] >= 0 ? 1 : 0));
        }
        return (fs / series.length) * zcr;
    }

    public static void main(String[] args) {
        AudioFeatures af = new AudioFeatures();
        Instances[] data = new Instances[2];
        data[0] = loadDataNullable("Z:/ArchiveData/Univariate_arff" + "/" + DatasetLists.tscProblems112[88] + "/"
                + DatasetLists.tscProblems112[88] + "_TRAIN");
        data[0].addAll(loadDataNullable("Z:/ArchiveData/Univariate_arff" + "/" + DatasetLists.tscProblems112[88] + "/"
                + DatasetLists.tscProblems112[88] + "_TEST"));
        System.out.println(data[0].relationName());
        try {
            data[1] = af.transform(data[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Before transform.
        System.out.println(data[0].get(0).toString());
        // After transform.
        for (int i = 0; i < data[1].size(); i++) {
            System.out.println(data[1].get(i).toString());
        }

    }


}
