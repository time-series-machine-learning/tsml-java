package timeseriesweka.filters;
import experiments.data.DatasetLists;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

import static experiments.data.DatasetLoading.loadDataNullable;
import static utilities.InstanceTools.fromWekaInstancesArray;

public class Fast_FFT extends SimpleBatchFilter{
    final String className = "sandbox.transforms.FFT";
    int nfft = 512;

    /**
     * Parses a given list of options. <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     * <pre> -L &lt;num&gt;
     *  max lag for the ACF function</pre>
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     * */
    @Override
    public void setOptions(String[] options) throws Exception {
        String nfftString=Utils.getOption('L', options);
        if (nfftString.length() != 0)
            nfft = Integer.parseInt(nfftString);
        else
            nfft = 512;
    }

    public void setNFFT(int nfft){
        this.nfft = nfft;
    }

    @Override
    public String globalInfo() {
        return null;
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        Instances instances = null;
        if(inputFormat.attribute("samplerate") != null){
            nfft = (int)inputFormat.get(0).value(inputFormat.attribute("samplerate"));
            inputFormat.deleteAttributeAt(inputFormat.attribute("samplerate").index());
        }else{
            nfft = inputFormat.numAttributes() - 1;
        }
        nearestPowerOF2(nfft);

        FastVector attributes = new FastVector(nfft/2);
        for (int i = 0; i < (nfft/2); i++) {
            attributes.addElement(new Attribute("FFT_att" + String.valueOf(i + 1)));
        }

        FastVector classValues = new FastVector(inputFormat.classAttribute().numValues());
        for(int i=0;i<inputFormat.classAttribute().numValues();i++)
            classValues.addElement(inputFormat.classAttribute().value(i));
        attributes.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),classValues));

        instances = new Instances("", attributes, 0);

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    private void nearestPowerOF2(int x){
        float power = (float)(Math.log(x) / Math.log(2));
        int m = (int)Math.ceil(power);
        nfft = (int)Math.pow(2.0, (double)m);
    }

    public Instances process(Instances input) throws Exception {
        Instances instances = new Instances(input);
        double[][] data = null;
        double[][] FFTData = null;
        Instances FFTInstances = null;
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] complexData = null;

        data = fromWekaInstancesArray(instances, true);
        FFTInstances = determineOutputFormat(instances);

        FFTData = new double[instances.size()][nfft/2 + 1];

        for (int i = 0; i < FFTData.length; i++){

            complexData = new Complex[nfft];
            for (int j = 0; j < complexData.length; j++){
                complexData[j] = new Complex(0.0, 0.0);
            }

            double mean = 0;
            if (data[i].length < nfft){
                for (int j = 0; j < data[i].length; j++) {
                    mean += data[i][j];
                }
                mean /= data[i].length;
            }

            //int limit = nfft < data[i].length ? nfft : data[i].length;
            for (int j = 0; j < nfft; j++) {
                if(j < data[i].length)
                    complexData[j] = new Complex(data[i][j], 0);
                else
                    complexData[j] = new Complex(mean, 0);
            }

            complexData = fft.transform(complexData, TransformType.FORWARD);

            for (int j = 0; j < (nfft/2); j++) {
                FFTData[i][j] = complexData[j].abs();
            }

            FFTData[i][FFTData[i].length - 1] = instances.get(i).classValue();
        }

        for (int i = 0; i < FFTData.length; i++) {
            FFTInstances.add(new DenseInstance(1.0, FFTData[i]));
        }

        return FFTInstances;
    }

    public static void main(String[] args) {
        Fast_FFT fast_fft = new Fast_FFT();
        Instances[] data = new Instances[2];
        data[0] = loadDataNullable(DatasetLists.beastPath + "TSCProblems" + "/" + DatasetLists.tscProblems85[28] + "/" + DatasetLists.tscProblems85[28]);
        try {
            data[1] = fast_fft.process(data[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        //Before transform.
        System.out.println(data[0].get(0).toString());
        //After transform.
        System.out.println(data[1].get(0).toString());
    }
}
