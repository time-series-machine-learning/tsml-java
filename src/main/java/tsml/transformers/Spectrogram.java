package tsml.transformers;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.*;

import static experiments.data.DatasetLoading.loadDataNullable;

import java.util.ArrayList;
import java.util.List;

public class Spectrogram implements Transformer {

    private int nfft = 256;
    private int windowLength = 75;
    private int overlap = 70;

    public Spectrogram() {
    }

    public Spectrogram(int windowLength, int overlap, int nfft) {
        this.windowLength = windowLength;
        this.overlap = overlap;
        this.nfft = nfft;
    }

    public void setNFFT(int x) {
        nfft = x;
    }

    public int getNFFT() {
        return nfft;
    }

    public void setWindowLength(int x) {
        windowLength = x;
    }

    public void setOverlap(int x) {
        overlap = x;
    }

    public String globalInfo() {
        return null;
    }

    public Instances determineOutputFormat(Instances inputFormat) {
        Instances instances = null;

        FastVector<Attribute> attributes = new FastVector<>(nfft / 2);
        for (int i = 0; i < (nfft / 2); i++) {
            attributes.addElement(new Attribute("Spec_att" + String.valueOf(i + 1)));
        }

        attributes.addElement(inputFormat.attribute(inputFormat.numAttributes() - 1));

        instances = new Instances("", attributes, 0);

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    @Override
    public Instance transform(Instance inst) {
        /*
         * double[] signal = new double[instances.get(i).numAttributes() - 1];
         * 
         * for (int j = 0; j < instances.get(i).numAttributes() - 1; j++) { signal[j] =
         * instances.get(i).value(j); } double[][] spectrogram = spectrogram(signal,
         * windowLength, overlap, nfft); Instances spectrogramsInstances =
         * MatrixToInstances(spectrogram, instances.classAttribute(),
         * instances.get(i).classValue()); return null;
         */

        // TODO: Not sure on how to convert this for a single instance.
        throw new NotImplementedException("This is not implemented for single transformation");
    }

    public Instances transform(Instances instances) {

        double[][] spectrogram = null;
        Instances[] spectrogramsInstances = new Instances[instances.numInstances()];
        double[] signal = null;
        for (int i = 0; i < instances.size(); i++) {
            signal = new double[instances.get(i).numAttributes() - 1];

            for (int j = 0; j < instances.get(i).numAttributes() - 1; j++) {
                signal[j] = instances.get(i).value(j);
            }
            spectrogram = spectrogram(signal, windowLength, overlap, nfft);
            spectrogramsInstances[i] = MatrixToInstances(spectrogram, instances.classAttribute(),
                    instances.get(i).classValue());
        }

        // Rearrange data
        Instances[] temp = new Instances[spectrogramsInstances[0].size()];
        for (int i = 0; i < temp.length; i++) {
            temp[i] = new Instances(spectrogramsInstances[0], 0);
            for (int j = 0; j < spectrogramsInstances.length; j++) {
                temp[i].add(spectrogramsInstances[j].get(i));
            }
        }

        return MultivariateInstanceTools.concatinateInstances(temp);
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        List<TimeSeries> out = new ArrayList<>();
        for (TimeSeries ts : inst) {
            double[] signal = ts.toValueArray();
            double [][] spectrogram = spectrogram(signal, windowLength, overlap, nfft);

            for(double[] spec : spectrogram){
                out.add(new TimeSeries(spec));
            }
        }
        return new TimeSeriesInstance(inst.getLabelIndex(), inst.getClassLabels(), out);
    }

    public int getNumWindows(int signalLength) {
        return (int) Math.floor((signalLength - overlap) / (windowLength - overlap));
    }

    private void checkParameters(int signalLength) {
        windowLength = windowLength < (int) (signalLength * 0.5) ? windowLength : (int) (signalLength * 0.5);
        overlap = overlap < (int) (windowLength * 0.5) ? overlap : (windowLength / 2);
    }

    public double[][] spectrogram(double[] signal, int windowWidth, int overlap, int nfft) {
        checkParameters(signal.length);
        int numWindows = getNumWindows(signal.length);
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        double[][] spectrogram = new double[numWindows][nfft / 2];
        Complex[] STFFT = null;
        for (int i = 0; i < numWindows; i++) {
            STFFT = new Complex[nfft];
            for (int j = 0; j < nfft; j++) {
                STFFT[j] = new Complex(0.0, 0.0);
            }
            for (int j = 0; j < windowLength; j++) {
                double temp = signal[j + (i * (this.windowLength - this.overlap))]
                        * (0.56 - 0.46 * Math.cos(2 * Math.PI * ((double) j / (double) this.windowLength)));
                STFFT[j] = new Complex(temp, 0.0);
            }
            STFFT = fft.transform(STFFT, TransformType.FORWARD);
            for (int j = 0; j < nfft / 2; j++) {
                spectrogram[i][j] = STFFT[j].abs();
            }
        }
        return spectrogram;
    }

    private Instances MatrixToInstances(double[][] data, Attribute classAttribute, double classValue) {
        Instances instances = null;

        FastVector<Attribute> attributes = new FastVector<>(data[0].length);
        for (int i = 0; i < data[0].length; i++) {
            attributes.addElement(new Attribute("attr" + String.valueOf(i + 1)));
        }
        attributes.addElement(classAttribute);

        instances = new Instances("", attributes, data.length);

        double[] temp = null;
        for (int i = 0; i < data.length; i++) {
            temp = new double[instances.numAttributes()];
            for (int j = 0; j < data[i].length; j++) {
                temp[j] = data[i][j];
            }
            temp[temp.length - 1] = classValue;
            instances.add(new DenseInstance(1.0, temp));
        }

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    public static void main(String[] args) {
        Spectrogram spec = new Spectrogram(75, 70, 256);
        Instances[] data = new Instances[2];

        data[0] = loadDataNullable(args[0] + args[1] + "/" + args[1] + "_TRAIN.arff");
        data[1] = loadDataNullable(args[0] + args[1] + "/" + args[1] + "_TEST.arff");
        System.out.println(data[0].get(0).toString());
        data[1] = spec.transform(data[0]);
        System.out.println();
        for (int i = 0; i < data[1].size(); i++) {
            Instance[] temp = MultivariateInstanceTools.splitMultivariateInstanceWithClassVal(data[1].get(i));
            System.out.println(temp[0]);
        }
        System.out.println();
        System.out.println("Signal length:" + (data[0].numAttributes() - 1));
        System.out.println("Window length: " + spec.windowLength);
        System.out.println("Overlap: " + spec.overlap);
        System.out.println("NFFT: " + spec.nfft);
    }




}
