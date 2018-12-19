package timeseriesweka.filters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.TreeSet;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.NormalizeCase;
import weka.filters.SimpleBatchFilter;

/**
 * Filter to transform time series into a bag of patterns representation.
 * i.e pass a sliding window over each series
 * normalise and convert each window to sax form
 * build a histogram of non-trivially matching patterns
 * 
 * Resulting in a bag (histogram) of patterns (SAX words) describing the high-level
 * structure of each timeseries
 *
 * Params: wordLength, alphabetSize, windowLength
 * 
 * @author James
 */
public class BagOfPatternsFilter extends SimpleBatchFilter {

    public TreeSet<String> dictionary;
    
    private final int windowSize;
    private final int numIntervals;
    private final int alphabetSize;
    private boolean useRealAttributes = true;
    
    private boolean numerosityReduction = false; //can expand to different types of nr
    //like those in senin implementation later, if wanted
    
    private FastVector alphabet = null;
    
    private static final long serialVersionUID = 1L;

    public BagOfPatternsFilter(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.numIntervals = PAA_intervalsPerWindow;
        this.alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        
        alphabet = SAX.getAlphabet(SAX_alphabetSize);
    }
    
    public int getWindowSize() {
        return numIntervals;
    }
    
    public int getNumIntervals() {
        return numIntervals;
    }

    public int getAlphabetSize() {
        return alphabetSize;
    }
    
    public void useRealValuedAttributes(boolean b){
        useRealAttributes = b;
    }
    
    public void performNumerosityReduction(boolean b){
        numerosityReduction = b;
    }
    
    private HashMap<String, Integer> buildHistogram(LinkedList<double[]> patterns) {
        
        HashMap<String, Integer> hist = new HashMap<>();

        for (double[] pattern : patterns) {   
            //convert to string                
            String word = "";
            for (int j = 0; j < pattern.length; ++j)
                word += (String) alphabet.get((int)pattern[j]);

            
            Integer val = hist.get(word);
            if (val == null)
                val = 0;
            
            hist.put(word, val+1);
        }
        
        return hist;
    }
    
    public HashMap<String, Integer> buildBag(Instance series) throws Exception {
       
        LinkedList<double[]> patterns = new LinkedList<>();
        
        double[] prevPattern = new double[windowSize];
        for (int i = 0; i < windowSize; ++i) 
            prevPattern[i] = -1;
        
        for (int windowStart = 0; windowStart+windowSize-1 < series.numAttributes()-1; ++windowStart) { 
            double[] pattern = slidingWindow(series, windowStart);
            
            try {
                NormalizeCase.standardNorm(pattern);
            } catch(Exception e) {
                //throws exception if zero variance
                //if zero variance, all values in window the same 
                for (int j = 0; j < pattern.length; ++j)
                    pattern[j] = 0;
            }
            pattern = SAX.convertSequence(pattern, alphabetSize, numIntervals);
            
            if (!(numerosityReduction && identicalPattern(pattern, prevPattern)))
                patterns.add(pattern);
        }
        
        return buildHistogram(patterns);
    }
    
    private double[] slidingWindow(Instance series, int windowStart) {
        double[] window = new double[windowSize];

        //copy the elements windowStart to windowStart+windowSize from data into the window
        for (int i = 0; i < windowSize; ++i)
            window[i] = series.value(i + windowStart);
        
        return window;
    }

    private boolean identicalPattern(double[] a, double[] b) {
        for (int i = 0; i < a.length; ++i)
            if (a[i] != b[i])
                return false;
        
        return true;
    }
  
    @Override
    protected Instances determineOutputFormat(Instances inputFormat)
            throws Exception {
        
        //Check all attributes are real valued, otherwise throw exception
        for (int i = 0; i < inputFormat.numAttributes(); i++) {
            if (inputFormat.classIndex() != i) {
                if (!inputFormat.attribute(i).isNumeric()) {
                    throw new Exception("Non numeric attribute not allowed for BoP conversion");
                }
            }
        }

        FastVector attributes = new FastVector();
        for (String word : dictionary) 
            attributes.add(new Attribute(word));
        
        Instances result = new Instances("BagOfPatterns_" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        
        if (inputFormat.classIndex() >= 0) {	//Classification set, set class 
            //Get the class values as a fast vector			
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.addElement(target.value(i));
            }
            
            result.insertAttributeAt(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals), result.numAttributes());
            result.setClassIndex(result.numAttributes() - 1);
        }
 
        return result;
    }

    @Override
    public String globalInfo() {
        return null;
    }

    @Override
    public Instances process(final Instances input) 
            throws Exception {
        
        ArrayList< HashMap<String, Integer> > bags = new ArrayList<>(input.numInstances());
        dictionary = new TreeSet<>();
        
        for (int i = 0; i < input.numInstances(); i++) {
            bags.add(buildBag(input.get(i)));
            dictionary.addAll(bags.get(i).keySet());
        }
        
        Instances output = determineOutputFormat(input); //now that dictionary is known, set up output
        
        Iterator<HashMap<String, Integer> > it = bags.iterator();
        int i = 0;
        while (it.hasNext()) {
            double[] bag = bagToArray(it.next());
            
            it.remove(); //freeing space asap, now that data is in array form as needed
            
            output.add(new SparseInstance(1.0, bag));
            output.get(i).setClassValue(input.get(i).classValue());
            
            ++i;
        }
        
        return output;
    }

    public double[] bagToArray(HashMap<String, Integer> bag) {
        double[] res = new double[dictionary.size()];
            
        int j = 0;
        for (String word : dictionary) {
            Integer val = bag.get(word);
            if (val != null)
                res[j] += val;
            ++j;
        }

        return res;
    }

    public String getRevision() {
        // TODO Auto-generated method stub
        return null;
    }

    public static void main(String[] args) {
        System.out.println("BoPtest\n\n");

        try {
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
            test.deleteAttributeAt(0); //just name of bottle
          
            BagOfPatternsFilter bop = new BagOfPatternsFilter(8,4,50);  
            bop.useRealValuedAttributes(false);
            Instances result = bop.process(test);
            
            System.out.println(result);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
}
