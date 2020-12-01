package tsml.transformers;
import experiments.data.DatasetLists;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.InstanceTools;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import weka.core.*;

import static experiments.data.DatasetLoading.loadDataNullable;

public class Fast_FFT implements Transformer {
    final String className = "sandbox.transforms.FFT";
    int nfft = 512;

    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start --> Valid options are:
     * <p/>
     * 
     * <pre>
     *  -L &lt;num&gt;
     *  max lag for the ACF function
     * </pre>
     * 
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String nfftString = Utils.getOption('L', options);
        if (nfftString.length() != 0)
            nfft = Integer.parseInt(nfftString);
        else
            nfft = 512;
    }

    public void setNFFT(int nfft) {
        this.nfft = nfft;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        Instances instances = null;
        if (inputFormat.attribute("samplerate") != null) {
            nfft = (int) inputFormat.get(0).value(inputFormat.attribute("samplerate"));
            inputFormat.deleteAttributeAt(inputFormat.attribute("samplerate").index());
        } else {
            // nfft = inputFormat.numAttributes() - 1;
        }
        // nearestPowerOF2(nfft);

        FastVector attributes = new FastVector(nfft / 2);
        for (int i = 0; i < (nfft / 2); i++) {
            attributes.addElement(new Attribute("FFT_att" + String.valueOf(i + 1)));
        }

        FastVector classValues = new FastVector(inputFormat.classAttribute().numValues());
        for (int i = 0; i < inputFormat.classAttribute().numValues(); i++)
            classValues.addElement(inputFormat.classAttribute().value(i));
        attributes.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), classValues));

        instances = new Instances("", attributes, 0);

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    private void nearestPowerOF2(int x) {
        float power = (float) (Math.log(x) / Math.log(2));
        int m = (int) Math.ceil(power);
        nfft = (int) Math.pow(2.0, (double) m);
    }

    @Override
    public Instance transform(Instance inst) {
        Complex[] complexData = new Complex[nfft];
        double[] data = InstanceTools.ConvertInstanceToArrayRemovingClassValue(inst);
        for (int j = 0; j < complexData.length; j++) {
            complexData[j] = new Complex(0.0, 0.0);
        }

        double mean = 0;
        if (data.length < nfft) {
            for (int j = 0; j < data.length; j++) {
                mean += data[j];
            }
            mean /= data.length;
        }

        // int limit = nfft < data[i].length ? nfft : data[i].length;
        for (int j = 0; j < nfft; j++) {
            if (j < data.length)
                complexData[j] = new Complex(data[j], 0);
            else
                complexData[j] = new Complex(mean, 0);
        }

        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        complexData = fft.transform(complexData, TransformType.FORWARD);

        double[] FFTData = new double[(nfft / 2) + (inst.classIndex() >= 0 ? 1 : 0)];
        for (int j = 0; j < (nfft / 2); j++) {
            FFTData[j] = complexData[j].abs();
        }

        if (inst.classIndex() >= 0)
            FFTData[FFTData.length - 1] = inst.classValue();

        return new DenseInstance(1, FFTData);
    }

    
    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][];
        int i = 0;
        for (TimeSeries ts : inst) {
            //TODO: make this NaN Safe. Mean is NaN safe but toArray isnt.
            out[i++] = calculate_FFT(ts.toValueArray(), TimeSeriesSummaryStatistics.mean(ts));
        }
        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels()); 
    }

    private double[] calculate_FFT(double[] data, double mean) {
        Complex[] complexData = new Complex[nfft];
        for (int j = 0; j < complexData.length; j++) {
            complexData[j] = new Complex(0.0, 0.0);
        }

        // int limit = nfft < data[i].length ? nfft : data[i].length;
        for (int j = 0; j < nfft; j++) {
            if (j < data.length)
                complexData[j] = new Complex(data[j], 0);
            else
                complexData[j] = new Complex(mean, 0);
        }

        
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        complexData = fft.transform(complexData, TransformType.FORWARD);

        double[] FFTData = new double[(nfft / 2)];
        for (int j = 0; j < (nfft / 2); j++) {
            FFTData[j] = complexData[j].abs();
        }

        return FFTData;
    }


    public static void main(String[] args) {
        Fast_FFT fast_fft = new Fast_FFT();
        Instances[] data = new Instances[2];
        data[0] = loadDataNullable("Z:/ArchiveData/Univariate_arff/" + DatasetLists.tscProblems85[28] + "/"
                + DatasetLists.tscProblems85[28]);
        data[1] = fast_fft.transform(data[0]);

        // Before transform.
        System.out.println(data[0].get(0).toString());
        // After transform.
        System.out.println(data[1].get(0).toString());
    }



}
