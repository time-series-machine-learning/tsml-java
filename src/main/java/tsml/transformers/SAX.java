/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.transformers;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

import java.util.ArrayList;
import java.util.List;

/**
 * Filter to reduce dimensionality of and discretise a time series into SAX
 * form, does not normalize, must be done separately if wanted.
 * 
 * Output attributes can be in two forms - discrete alphabet or real values 0 to
 * alphabetsize-1
 * 
 * Default number of intervals = 8 Default alphabet size = 4
 *
 * @author James
 */
public class SAX implements Transformer, TechnicalInformationHandler {

    private int numIntervals = 8;
    private int alphabetSize = 4;
    private boolean useRealAttributes = false;
    private List<String> alphabet = null;

    private Instances inputFormat;

    private static final long serialVersionUID = 1L;

    // individual strings for each symbol in the alphabet, up to ten symbols
    private static final String[] alphabetSymbols = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" };

    public int getNumIntervals() {
        return numIntervals;
    }

    public int getAlphabetSize() {
        return alphabetSize;
    }

    public List<String> getAlphabet() {
        if (alphabet == null)
            generateAlphabet();
        return alphabet;
    }

    public static List<String> getAlphabet(int alphabetSize) {
        List<String> alphb = new ArrayList<>();
        for (int i = 0; i < alphabetSize; ++i)
            alphb.add(alphabetSymbols[i]);
        return alphb;
    }

    public void setNumIntervals(int intervals) {
        numIntervals = intervals;
    }

    public void setAlphabetSize(int alphasize) {

        if (alphasize > 10) {
            alphasize = 10;
            System.out.println("Alpha size too large, setting to 10");
        } else if (alphasize < 2) {
            alphasize = 2;
            System.out.println("Alpha size too small, setting to 2");
        }

        alphabetSize = alphasize;
    }

    public void useRealValuedAttributes(boolean b) {
        useRealAttributes = b;
    }

    public void generateAlphabet() {
        alphabet = new ArrayList<>();
        for (int i = 0; i < alphabetSize; ++i)
            alphabet.add(alphabetSymbols[i]);
    }

    // lookup table for the breakpoints for a gaussian curve where the area under
    // curve T between Ti and Ti+1 = 1/a, 'a' being the size of the alphabet.
    // columns up to a=10 stored
    // lit. suggests that a = 3 or 4 is bet in almost all cases, up to 6 or 7 at
    // most
    // for specific datasets
    public double[] generateBreakpoints(int alphabetSize) {
        double maxVal = Double.MAX_VALUE;
        double[] breakpoints = null;

        switch (alphabetSize) {
            case 2: {
                breakpoints = new double[] { 0, maxVal };
                break;
            }
            case 3: {
                breakpoints = new double[] { -0.43, 0.43, maxVal };
                break;
            }
            case 4: {
                breakpoints = new double[] { -0.67, 0, 0.67, maxVal };
                break;
            }
            case 5: {
                breakpoints = new double[] { -0.84, -0.25, 0.25, 0.84, maxVal };
                break;
            }
            case 6: {
                breakpoints = new double[] { -0.97, -0.43, 0, 0.43, 0.97, maxVal };
                break;
            }
            case 7: {
                breakpoints = new double[] { -1.07, -0.57, -0.18, 0.18, 0.57, 1.07, maxVal };
                break;
            }
            case 8: {
                breakpoints = new double[] { -1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, maxVal };
                break;
            }
            case 9: {
                breakpoints = new double[] { -1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, maxVal };
                break;
            }
            case 10: {
                breakpoints = new double[] { -1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28, maxVal };
                break;
            }
        }

        return breakpoints;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        ArrayList<Attribute> attributes = new ArrayList<>();

        // If the alphabet is to be considered as discrete values (i.e non real),
        // generate nominal values based on alphabet size
        if (!useRealAttributes)
            generateAlphabet();

        Attribute att;
        String name;

        for (int i = 0; i < numIntervals; i++) {
            name = "SAXInterval_" + i;

            if (!useRealAttributes)
                att = new Attribute(name, alphabet);
            else
                att = new Attribute(name);

            attributes.add(att);
        }

        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList<>();
            for (int i = 0; i < target.numValues(); i++) {
                vals.add(target.value(i));
            }
            attributes.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }

        Instances result = new Instances("SAX" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    @Override
    public Instances transform(Instances data) {
        Instances output = determineOutputFormat(data);

        // Convert input to PAA format
        PAA paa = new PAA();
        paa.setNumIntervals(numIntervals);
        // Now convert PAA -> SAX
        for (Instance inst : data) {
            // lower mem to do it series at a time.
            output.add(transform(paa.transform(inst)));
        }

        return output;
    }

    @Override
    public Instance transform(Instance inst) {
        double[] data = inst.toDoubleArray();

        // remove class attribute if needed
        double[] temp;
        int c = inst.classIndex();
        if (c >= 0) {
            temp = new double[data.length - 1];
            System.arraycopy(data, 0, temp, 0, c); // assumes class attribute is in last index
            data = temp;
        }

        convertSequence(data);

        // Now in SAX form, extract out the terms and set the attributes of new instance
        Instance newInstance = new DenseInstance(numIntervals + (inst.classIndex() >= 0 ? 1 : 0));

        for (int j = 0; j < numIntervals; j++)
            newInstance.setValue(j, data[j]);

        if (inst.classIndex() >= 0)
            newInstance.setValue(newInstance.numAttributes()-1, inst.classValue());

        return newInstance;
    }

    @Override
    public TimeSeriesInstances transform(TimeSeriesInstances data) {
        // Convert input to PAA format
        PAA paa = new PAA();
        paa.setNumIntervals(numIntervals);
        // Now convert PAA -> SAX

        TimeSeriesInstances out = new TimeSeriesInstances(data.getClassLabels());
        for (TimeSeriesInstance inst : data) {
            // lower mem to do it series at a time.
            out.add(transform(paa.transform(inst)));
        }

        return out;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        
        double[][] out = new double[inst.getNumDimensions()][];
        int i =0;
        for(TimeSeries ts : inst){
            double[] o = ts.toValueArray();
            convertSequence(o);
            out[i++] = o;
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    /**
     * converts a double[] of 'paa-ed' data to sax letters
     * 
     * @param data
     * @throws Exception
     */
    public void convertSequence(double[] data) {
        double[] gaussianBreakpoints = generateBreakpoints(alphabetSize);

        for (int i = 0; i < numIntervals; ++i) {
            // find symbol corresponding to each mean
            for (int j = 0; j < alphabetSize; ++j)
                if (data[i] < gaussianBreakpoints[j]) {
                    data[i] = j;
                    break;
                }
        }
    }

    /**
     * Will perform a SAX transformation on a single series passed as a double[]
     * 
     * @param alphabetSize size of SAX alphabet
     * @param numIntervals size of resulting word
     * @throws Exception
     */
    public static double[] convertSequence(double[] data, int alphabetSize, int numIntervals) {
        SAX sax = new SAX();
        sax.setNumIntervals(numIntervals);
        sax.setAlphabetSize(alphabetSize);
        sax.useRealValuedAttributes(true);

        double[] d = PAA.convertInstance(data, numIntervals);
        sax.convertSequence(d);

        return d;
    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet."); // To change body of generated methods, choose
                                                                       // Tools | Templates.
    }

    public static void main(String[] args) {
        System.out.println("SAXtest\n\n");

        Instances test = DatasetLoading
                .loadData("C:\\Users\\ajb\\Dropbox\\Data\\TSCProblems\\Chinatown\\Chinatown_TRAIN.arff");

        test = new RowNormalizer().transform(test);

        SAX sax = new SAX();

        sax.setNumIntervals(8);
        sax.setAlphabetSize(4);
        sax.useRealValuedAttributes(false);
        Instances result = sax.transform(test);

        System.out.println(test);
        System.out.println("\n\n\nResults:\n\n");
        System.out.println(result);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.TITLE,
                "Experiencing SAX: a novel symbolic representation of time series");
        result.setValue(TechnicalInformation.Field.AUTHOR, "Jessica Lin, Eamonn Keogh, Li Wei and Stefano Lonardi");
        result.setValue(TechnicalInformation.Field.YEAR, "2007");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "15");
        result.setValue(TechnicalInformation.Field.NUMBER, "2");
        result.setValue(TechnicalInformation.Field.PAGES, "107-144");

        return result;
    }

    
}
