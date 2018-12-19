/*
Code to generate results for the paper
Lines, Jason and Bagnall, Anthony (2015) Time series classification with 
ensembles of elastic distance measures. Data Mining and Knowledge Discovery Journal

@article{lines15elastic,
  title={Time Series Classification with Ensembles of Elastic Distance Measures},
  author={J. Lines and A. Bagnall},
  journal={Data Mining and Knowledge Discovery},
  volume={29},
  issue={3},
  pages={565--592},
  year={2015}
}

also published in 
Lines, Jason and Bagnall, Anthony (2014) Ensembles of Elastic Distance 
easures for Time Series Classification. In: Proceedings of SDM 2014
@inproceedings{lines14elastic,
	Author = {J. Lines and A. Bagnall},
	Title ="Ensembles of Elastic Distance Measures for Time Series Classification",
	Booktitle ="Proceedings of the 14th {SIAM} International Conference on Data Mining (SDM)",
	Year = {2014}
}


 */
package papers;

import timeseriesweka.classifiers.ElasticEnsemble;
import development.DataSets;
import fileIO.OutFile;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import weka.core.Instances;


public class Lines15elastic {
    //CHANGE THIS
    static String path="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";

/** This will be fairly slow! The EE is threaded for cross validation,but
 this also makes it memory hungry.
 
 * In reality, we decomposed the ensemble components and ran them concurrently. 
 * However, this approach is clearer, it does work and is sufficicent for small 
 * problems
 */    
//    public static double singleProblem(String problem, ElasticEnsemble.EnsembleType e) throws Exception{ // note: option of ensemble type removed as this is no longer supported in EE - only proportional scheme types are supported
    public static double singleProblem(String problem) throws Exception{
        ElasticEnsemble ee= new ElasticEnsemble();
        Instances train = ClassifierTools.loadData(path+problem+"\\"+problem+"_TRAIN");
        ee.buildClassifier(train);
        Instances test = ClassifierTools.loadData(path+problem+"\\"+problem+"_TEST");
        double a=ClassifierTools.accuracy(test, ee);
        return a;
    }
    
    public static double[] smallUCRProblems() throws Exception{
        OutFile of = new OutFile(path+"SmallUCRProblems.csv"); 
        double[] acc = new double[DataSets.tscProblemsSmall.length];
        DecimalFormat df = new DecimalFormat("##.###");
        for(int i=0;i<DataSets.tscProblemsSmall.length;i++){
//            acc[i]=singleProblem(DataSets.tscProblemsSmall[i],ElasticEnsemble.EnsembleType.Equal);
            acc[i]=singleProblem(DataSets.tscProblemsSmall[i]);
            System.out.println(DataSets.tscProblemsSmall[i]+" Error = "+df.format(1-acc[i]));
        }
        return acc;
    }
/* All problems: with no CV
    */
    public static double[] allProblems() throws Exception{
        OutFile of = new OutFile(path+"SmallUCRProblems.csv"); 
        double[] acc = new double[DataSets.tscProblemsSmall.length];
        DecimalFormat df = new DecimalFormat("##.###");
        for(int i=0;i<DataSets.tscProblems85.length;i++){
//            acc[i]=singleProblem(DataSets.tscProblems85[i],ElasticEnsemble.EnsembleType.Equal);
            acc[i]=singleProblem(DataSets.tscProblems85[i]);
            System.out.println(DataSets.tscProblems85[i]+" Error = "+df.format(1-acc[i]));
            of.writeLine(DataSets.tscProblems85[i]+","+(1-acc[i]));
        }
        return acc;
    }
    
    
    public static void main(String[] args) throws Exception{
/* Single Problem
        String prob="ItalyPowerDemand";
        double a=singleProblem(prob);
        System.out.println(" EE Train/Test Acc = "+a);
*/
 //       smallUCRProblems();
        allProblems();
    }
    
    
}
