package timeseriesweka.filters;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

import static experiments.data.DatasetLoading.loadDataNullable;

public class Spectrogram extends SimpleBatchFilter {

    private int nfft = 256;
    private int windowLength = 75;
    private int overlap = 70;

    public Spectrogram(){}

    public Spectrogram(int windowLength, int overlap, int nfft){
        this.windowLength = windowLength;
        this.overlap = overlap;
        this.nfft = nfft;
    }

    public void setNFFT(int x){nfft = x;}
    public int getNFFT(){return nfft;}
    public void setWindowLength(int x){windowLength = x;}
    public void setOverlap(int x){overlap = x;}

    public String globalInfo() {
        return null;
    }

    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        Instances instances = null;

        FastVector attributes = new FastVector(nfft/2);
        for (int i = 0; i < (nfft/2); i++) {
            attributes.addElement(new Attribute("Spec_att" + String.valueOf(i + 1)));
        }

        attributes.addElement(inputFormat.attribute(inputFormat.numAttributes() - 1));

        instances = new Instances("", attributes, 0);

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    public Instances process(Instances instances){

        double[][] spectrogram = null;
        Instances[] spectrogramsInstances = new Instances[instances.numInstances()];
        double[] signal = null;
        for (int i = 0; i < instances.size(); i++) {
            signal = new double[instances.get(i).numAttributes() - 1];

            for (int j = 0; j < instances.get(i).numAttributes() - 1; j++) {
                signal[j] = instances.get(i).value(j);
            }

            spectrogram = spectrogram(signal, windowLength, overlap, nfft);
            spectrogramsInstances[i] = MatrixToInstances(spectrogram, instances.classAttribute(), instances.get(i).classValue());
        }
        return MultivariateInstanceTools.mergeToMultivariateInstances(spectrogramsInstances);
    }

    public int getNumWindows(int signalLength){
        return (int)Math.floor((signalLength - overlap)/(windowLength - overlap));
    }

    public double[][] spectrogram(double[] signal, int windowWidth, int overlap, int nfft){
        int numWindows = getNumWindows(signal.length);
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        double[][] spectrogram = new double[numWindows][nfft/2];
        Complex[] STFFT = null;
        for (int i = 0; i < numWindows; i++) {
            STFFT = new Complex[nfft];
            for (int j = 0; j < nfft; j++) {
                STFFT[j] = new Complex(0.0, 0.0);
            }
            for (int j = 0; j < windowWidth; j++) {
                double temp = signal[j + (i * (windowWidth - overlap))]*(0.56 - 0.46*Math.cos(2*Math.PI*((double)j/(double)windowWidth)));
                STFFT[j] = new Complex(temp, 0.0);
            }
            STFFT = fft.transform(STFFT, TransformType.FORWARD);
            for (int j = 0; j < nfft/2; j++) {
                spectrogram[i][j] = STFFT[j].abs();
            }
        }
        return spectrogram;
    }

    private Instances MatrixToInstances(double[][] data, Attribute classAttribute, double classValue){
        Instances instances = null;

        FastVector attributes = new FastVector(data[0].length);
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

        data[0] = loadDataNullable(args[0] + args[1] + "/" + args[1] + ".arff");
        System.out.println(data[0].get(0).toString());
        data[1] = spec.process(data[0]);
        System.out.println();
        for (int i = 0; i < data[1].size(); i++) {
            Instance[] temp = MultivariateInstanceTools.splitMultivariateInstanceWithClassVal(data[1].get(i));
            System.out.println(temp[0]);
        }

    }
}
