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
package tsml.classifiers.dictionary_based;

import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import utilities.ClassifierTools;
import machine_learning.classifiers.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import tsml.transformers.SAX;
import weka.filters.unsupervised.instance.Randomize;

import java.util.concurrent.TimeUnit;

/**
 *
 * @author James
 */
public class SAX_1NN extends EnhancedAbstractClassifier {

    public Instances SAXdata;
    private kNN knn;
    private SAX sax;
    
    private final int PAA_intervalsPerWindow;
    private final int SAX_alphabetSize;
    
    public SAX_1NN(int PAA_intervalsPerWindow, int SAX_alphabetSize) { 
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        
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
        long startTime=System.nanoTime();
        
        SAXdata = sax.transform(data);
        knn.buildClassifier(SAXdata);

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime()-startTime);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return knn.classifyInstance(sax.transform(instance));
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return knn.distributionForInstance(sax.transform(instance));
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args) throws Exception{
        
//        System.out.println(ClassifierTools.testUtils_getIPDAcc(new SAX_1NN(10, 4)));
//        System.out.println(ClassifierTools.testUtils_confirmIPDReproduction(new SAX_1NN(10, 4), 0.9154518950437318, "2019_09_26"));
        
        System.out.println("BagofPatternsTest\n\n");
        
        try {
            Instances all = DatasetLoading.loadDataNullable("C:\\\\Temp\\\\TESTDATA\\\\FiveClassV1.arff");
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
