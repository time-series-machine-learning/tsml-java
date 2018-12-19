
package statistics.simulators;

import development.DataSets;
import fileIO.OutFile;
import java.text.DecimalFormat;
import timeseriesweka.classifiers.FastDTW_1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.filters.MatrixProfile;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class SimulateMatrixProfileData {
    static DataSimulator sim;
    static boolean debug=true;
    public static Instances generateMatrixProfileData(int seriesLength, int []casesPerClass)
    {
        MatrixProfileModelVersion1.setGlobalSeriesLength(seriesLength);
       
        MatrixProfileModelVersion1[] MP_Mod = new MatrixProfileModelVersion1[casesPerClass.length];
        populateMatrixProfileModels(MP_Mod); 
        sim = new DataSimulator(MP_Mod);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;
    }
    private static void populateMatrixProfileModels(MatrixProfileModelVersion1[] m){
        if(m.length!=2)
            System.out.println("ONLY IMPLEMENTED FOR TWO CLASSES");
//Create two models with same interval but different shape. 
        MatrixProfileModelVersion1 m1=new MatrixProfileModelVersion1();
        MatrixProfileModelVersion1 m2=new MatrixProfileModelVersion1();
        
        m[0]=m1;
        m[1]=m2;
        if(debug){
            System.out.println(" Model 1 = "+m[0]);
            System.out.println(" Model 2 = "+m[1]);
            
        }
            
        
    }
    
    private static void test1NNClassifiers() throws Exception{
        for(double sig=0;sig<=1;sig+=0.5){
            Model.setDefaultSigma(sig);
            double meanAcc=0;
            double meanAcc2=0;
            double meanAcc3=0;
            int r=100;
            DecimalFormat df= new DecimalFormat("###.#####");
            for(int i=0;i<r;i++){
                Model.setGlobalRandomSeed(i);
                int seriesLength=150;
                int[] casesPerClass=new int[]{50,50};        
                Instances d=generateMatrixProfileData(seriesLength,casesPerClass);
                if(i==1){
                    OutFile out=new OutFile("C:\\temp\\mpRand"+sig+".csv");
                    out.writeString(d.toString());
                }
                Instances[] split=InstanceTools.resampleInstances(d,i,0.1);
                kNN knn= new kNN();
                knn.setKNN(1);
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(knn, split[0], split[1]);

                NormalizeCase nc=new NormalizeCase();
                split[0]=nc.process(split[0]);
                split[1]=nc.process(split[1]);
                double acc2=ClassifierTools.singleTrainTestSplitAccuracy(knn, split[0], split[1]);
                MatrixProfile mp=new MatrixProfile(29);
                Instances[] mpSplit=new Instances[2];
                mpSplit[0]=mp.process(split[0]);
                mpSplit[1]=mp.process(split[1]);
                double acc3=ClassifierTools.singleTrainTestSplitAccuracy(knn, mpSplit[0], mpSplit[1]);
                meanAcc+=acc;
                meanAcc2+=acc2;
                meanAcc3+=acc3;
                System.out.println("Train Size ="+split[0].numInstances()+" 1NN acc = "+df.format(acc)+" 1NN acc normed="+df.format(acc2)+" 1NN MP acc ="+df.format(acc3));
            }
            System.out.println(" Sig ="+sig+" Mean 1NN Acc ="+df.format(meanAcc/r)+" Mean 1NN Norm Acc ="+df.format(meanAcc2/r)+" Mean 1NN MP Acc = "+df.format(meanAcc3/r));
        }
        
    }
 
    public static void createExampleData() throws Exception{
        OutFile raw=new OutFile("C:\\Temp\\raw.csv");
        OutFile mpFile=new OutFile("C:\\Temp\\mp.csv");
        Model.setDefaultSigma(1.0);
        DecimalFormat df= new DecimalFormat("###.#####");
        Model.setGlobalRandomSeed(1);
        int seriesLength=150;
        int[] casesPerClass=new int[]{50,50};        
        Instances d=generateMatrixProfileData(seriesLength,casesPerClass);
        MatrixProfile mp=new MatrixProfile(29);
        Instances md;
        md=mp.process(d);
        raw.writeLine(d.toString());
        mpFile.writeLine(md.toString());
    }
    public static void main(String[] args) throws Exception {
        createExampleData();
        System.exit(0);
        test1NNClassifiers();
        
        Model.setDefaultSigma(1);
        Model.setGlobalRandomSeed(0);
        int seriesLength=500;
        int[] casesPerClass=new int[]{100,100};        
        NormalizeCase nc=new NormalizeCase();
        Instances d=generateMatrixProfileData(seriesLength,casesPerClass);
        Instances[] split=InstanceTools.resampleInstances(d, 0,0.1);
        OutFile of = new OutFile("C:\\Temp\\train.arff");
        of.writeString(split[0].toString()+"");
        of = new OutFile("C:\\Temp\\test.arff");
        of.writeString(split[1].toString()+"");
        MatrixProfile mp=new MatrixProfile(29);
        Instances m1=mp.process(split[0]);
//        m1=nc.process(m1);
        of = new OutFile("C:\\Temp\\MPTrain.arff");
        of.writeString(split[0]+"");
        Instances m2=mp.process(split[1]);
 //       m2=nc.process(m2);
        of = new OutFile("C:\\Temp\\MPTest.arff");
        of.writeString(split[1].toString()+"\n\n");
        of.writeString(m2.toString());
    }
    
    
    
}
