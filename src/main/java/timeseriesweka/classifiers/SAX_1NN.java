
package timeseriesweka.classifiers;

import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import timeseriesweka.filters.SAX;
import weka.filters.unsupervised.instance.Randomize;

/**
 *
 * @author James
 */
public class SAX_1NN extends AbstractClassifierWithTrainingData {

    public Instances SAXdata;
    private kNN knn;
    private SAX sax;
    
    private final int PAA_intervalsPerWindow;
    private final int SAX_alphabetSize;
    
    public SAX_1NN(int PAA_intervalsPerWindow, int SAX_alphabetSize) { 
        this.PAA_intervalsPerWindow = PAA_intervalsPerWindow;
        this.SAX_alphabetSize = SAX_alphabetSize;
        
        sax = new SAX();
        sax.setNumIntervals(PAA_intervalsPerWindow);
        sax.setAlphabetSize(SAX_alphabetSize); 
        sax.useRealValuedAttributes(false);
        
        knn = new kNN(); //default to 1NN, Euclidean distance
    }
    @Override
    public String getParameters() {
        return super.getParameters()+",PAAIntervalsPerWindow,"+PAA_intervalsPerWindow+",alphabetSize,"+SAX_alphabetSize;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainResults.buildTime=System.currentTimeMillis();
        
        SAXdata = sax.process(data);
        knn.buildClassifier(SAXdata);
        trainResults.buildTime=System.currentTimeMillis()-trainResults.buildTime;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instance saxinst = sax.convertInstance(instance, SAX_alphabetSize, PAA_intervalsPerWindow);
        return knn.classifyInstance(saxinst);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance saxinst = sax.convertInstance(instance, SAX_alphabetSize, PAA_intervalsPerWindow);
        return knn.distributionForInstance(saxinst);
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        System.out.println("BagofPatternsTest\n\n");
        
        try {
            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\FiveClassV1.arff");
            all.deleteAttributeAt(0); //just name of bottle        
            
            Randomize rand = new Randomize();
            rand.setInputFormat(all);
            for (int i = 0; i < all.numInstances(); ++i) {
                rand.input(all.get(i));
            }
            rand.batchFinished();
            
            int trainNum = (int) (all.numInstances() * 0.7);
            int testNum = all.numInstances() - trainNum;
            
            Instances train = new Instances(all, trainNum);
            for (int i = 0; i < trainNum; ++i) 
                train.add(rand.output());
            
            Instances test = new Instances(all, testNum);
            for (int i = 0; i < testNum; ++i) 
                test.add(rand.output());
            
            SAX_1NN saxc = new SAX_1NN(6,3);
            saxc.buildClassifier(train);
            
            System.out.println(saxc.SAXdata);
            
            System.out.println("\nACCURACY TEST");
            System.out.println(ClassifierTools.accuracy(test, saxc));

        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        
    }
    
    @Override
    public String toString() { 
        return "SAX";
    }
    
}
