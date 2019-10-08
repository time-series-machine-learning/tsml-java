package timeseriesweka.filters;;
import org.apache.commons.math3.transform.DctNormalization;
import org.apache.commons.math3.transform.FastCosineTransformer;
import org.apache.commons.math3.transform.TransformType;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

import static experiments.data.DatasetLoading.loadDataNullable;

public class MFCC{

    //Default values (samples), assuming no infile info on samplerate.
    int windowLength = 75;
    int overlapLength = 70;
    //Default values (miliseconds), determined using infile info on samplerate.
    int windowDuration = 25;
    int overlapDuration = 10;

    Boolean checkForSampleRate = true;
    int nfft = 2048;
    int sampleRate = 44100;
    Spectrogram spectrogram;
    FastCosineTransformer dct = new FastCosineTransformer(DctNormalization.STANDARD_DCT_I);
    int numFilterBanks = 33;
    double[][] filterBank = null;
    double[][] melFreqCepsCo = null;
    //Upper and lower frequencies the filter bank will be applied to (Freq. outside of these will not contribute to overall output.).
    int lowerFreq = 100;
    int upperFreq = 2000;


    public MFCC(){
        spectrogram = new Spectrogram(windowLength, overlapLength, nfft);
    }

    public String globalInfo() {
        return null;
    }

    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        Instances instances = null;
        FastVector attributes = new FastVector(numFilterBanks);
        for (int i = 0; i < (numFilterBanks); i++) {
            attributes.addElement(new Attribute("MFCC_att" + String.valueOf(i + 1)));
        }

        FastVector classValues = new FastVector(inputFormat.classAttribute().numValues());
        for(int i=0;i<inputFormat.classAttribute().numValues();i++)
            classValues.addElement(inputFormat.classAttribute().value(i));
        attributes.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),classValues));

        instances = new Instances("", attributes, 0);

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    public Instances process(Instance instance) throws Exception {
        Instances MFCCInstances = null;
        double[][] spectrogram = null;
        double cumalativeFilteredVals = 0;
        double[] signal = new double[instance.numAttributes() - 1];

        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            signal[i] = instance.value(i);
        }

        if ((instance.dataset().attribute("samplerate") != null) && checkForSampleRate) {
            sampleRate = (int) instance.dataset().get(0).value(instance.dataset().attribute("samplerate"));
            instance.dataset().deleteAttributeAt(instance.dataset().attribute("samplerate").index());
        }
        if(sampleRate == 0){
            sampleRate = nfft;
        }

        windowLength = (int)((windowDuration/1000.0) * (double)sampleRate);
        overlapLength = (int)((overlapDuration/1000.0) * (double)sampleRate);
        this.spectrogram.setWindowLength(windowLength);
        this.spectrogram.setOverlap(overlapLength);

        spectrogram = this.spectrogram.spectrogram(signal, windowLength, overlapLength, nfft);

        //Performed to create Periodogram estimate of the power spectrum.
        for (int i = 0; i < spectrogram.length; i++) {
            for (int j = 0; j < spectrogram[i].length; j++) {
                spectrogram[i][j] = (1/(double)spectrogram[i].length)* Math.pow(spectrogram[i][j], 2);
            }
        }

        filterBank = createFilterBanks();
        melFreqCepsCo = new double[spectrogram.length][filterBank.length];

        for (int i = 0; i < spectrogram.length; i++) {
            for (int j = 0; j < filterBank.length; j++) {
                cumalativeFilteredVals = 0;
                for (int k = 0; k < spectrogram[i].length; k++) {
                    cumalativeFilteredVals += spectrogram[i][k] * filterBank[j][k];
                }
                melFreqCepsCo[i][j] = Math.log(cumalativeFilteredVals);
            }
        }

        for (int i = 0; i < melFreqCepsCo.length; i++) {
            melFreqCepsCo[i] = dct.transform(melFreqCepsCo[i], TransformType.FORWARD);
        }

        MFCCInstances = determineOutputFormat(instance.dataset());
        double[] temp;
        for (int i = 0; i < melFreqCepsCo.length; i++) {
            temp = new double[numFilterBanks + 1];
            for (int j = 0; j < numFilterBanks; j++) {
                temp[j] = melFreqCepsCo[i][j];
            }
            temp[temp.length - 1] = instance.value(instance.numAttributes() - 1);
            MFCCInstances.add(new DenseInstance(1.0, temp));
        }
        return MFCCInstances;
    }

    private double[][] createFilterBanks(){
        filterBank = new double[numFilterBanks][nfft/2];
        double[] filterPeaks = new double[numFilterBanks + 2];
        //Local overload for holding Mel conversion.
        double lowerFreq = 1125 * Math.log(1 + (this.lowerFreq / (double)700));
        double upperFreq = 1125 * Math.log(1 + (this.upperFreq / (double)700));
        double step = (upperFreq - lowerFreq) / (filterPeaks.length - 1);
        
        for (int i = 0; i < filterPeaks.length; i++) {
            filterPeaks[i] = lowerFreq + (step * i);
        }

        //Back to hertz.
        for (int i = 0; i < filterPeaks.length; i++) {
            filterPeaks[i] = 700 * (Math.exp(filterPeaks[i] / 1125) - 1);
        }

        for (int i = 0; i < filterPeaks.length; i++) {
            filterPeaks[i] = Math.floor((nfft + 1) * (filterPeaks[i] / this.sampleRate));
        }

        //Create Filter Banks.
        for (int i = 0; i < filterBank.length; i++) {
            for (int j = 0; j < filterBank[i].length; j++) {
                if(j < filterPeaks[i]){
                    filterBank[i][j] = 0;
                }
                if(j >= filterPeaks[i] && j <= filterPeaks[i + 1]){
                    filterBank[i][j] = ((j - filterPeaks[i])/(filterPeaks[i + 1] - filterPeaks[i]));
                }
                if(j > filterPeaks[i + 1] && j < filterPeaks[i + 2]){
                    filterBank[i][j] = ((filterPeaks[i + 2] - j) / (filterPeaks[i + 2] - filterPeaks[i + 1]));
                }
                if(j > filterPeaks[i + 2]){
                    filterBank[i][j] = 0;
                }
            }
        }

        return filterBank;
    }

    public static void main (String[]args){
        MFCC mfcc = new MFCC();
        Instances[] data = new Instances[2];
        data[0] = loadDataNullable("D:\\Test\\Datasets\\Truncated\\DuckDuckGeese\\DuckDuckGeese");
        System.out.println(data[0].get(0).toString());
        try {
            data[1] = mfcc.process(data[0].instance(0));
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println();
        for (int i = 0; i < data[1].size(); i++) {
            System.out.println(data[1].get(i).toString());
        }
    }
}
