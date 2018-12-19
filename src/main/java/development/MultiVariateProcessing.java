/*
Multivariate data can be stored in Wekas "multi instance" format
https://weka.wikispaces.com/Multi-instance+classification

for TSC, the basic univariate syntax is 

 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Sorting out the new archive
 * @author ajb
 */
public class MultiVariateProcessing {
    
    public static void makeConcatenatedFiles(){
        String path="Z:\\Data\\Multivariate TSC Problems\\";
        String dest="Z:\\Data\\ConcatenatedMTSC\\";
        OutFile out=new OutFile(path+"SummaryData.csv");
        out.writeLine("problem,numTrainCases,numTestCases,numDimensions,seriesLength,numClasses");
        String[] probs={"BasicMotions"};
        for(String prob:DataSets.mtscProblems2018){
            File t1=new File(dest+prob+"\\"+prob+"_TRAIN.arff");
            File t2=new File(dest+prob+"\\"+prob+"_TRAIN.arff");
            if(!(t1.exists()||t2.exists())){
                Instances train =ClassifierTools.loadData(path+prob+"\\"+prob+"_TRAIN");
                Instances test =ClassifierTools.loadData(path+prob+"\\"+prob+"_TEST");
                System.out.println("PROBLEM "+prob);        
                System.out.println("Num train instances ="+train.numInstances());
                System.out.println("Num test instances ="+test.numInstances());
                System.out.println("num train attributes (should be 2!)="+train.numAttributes());
                System.out.println("num classes="+train.numClasses());
                Instance temp=train.instance(0);
                Instances x= temp.relationalValue(0);
                System.out.println("train number of dimensions "+x.numInstances());
                System.out.println("train number of attributes per dimension "+x.numAttributes());
                 temp=test.instance(0);
                x= temp.relationalValue(0);
                System.out.println("test number of dimensions "+x.numInstances());
                System.out.println("test number of attributes per dimension "+x.numAttributes());
                out.writeLine(prob+","+train.numInstances()+","+test.numInstances()+","+x.numInstances()+","+x.numAttributes()+","+train.numClasses());
                
                int numAtts=x.numInstances()*x.numAttributes();
                System.out.println(" Total number of attributes ="+numAtts);
    //Build a new train test file of concatenated attributes
                File f= new File(dest+prob);
                f.mkdirs();
                OutFile uniTrain=new OutFile(dest+prob+"\\"+prob+"_TRAIN.arff");
                OutFile uniTest=new OutFile(dest+prob+"\\"+prob+"_TEST.arff");;
                String header="@relation "+prob+"\n";
                for(int i=0;i<numAtts;i++){
                    header+="@attribute att"+i+" numeric \n";
                }
                header+="@attribute "+train.classAttribute().name()+ " {";
                for(int i=0;i<train.numClasses()-1;i++)
                    header+=i+",";
                header+=train.numClasses()-1+"}\n";
                header+="@data \n";
                uniTrain.writeString(header);
                uniTest.writeString(header);
                for(int i=0;i<train.numInstances();i++){
                    temp=train.instance(i);
                    x= temp.relationalValue(0);
                    for(Instance y:x){//Each dimension
                        for(int j=0;j<y.numAttributes();j++)
                            uniTrain.writeString(y.value(j)+",");
                    }
                    uniTrain.writeString((int)temp.classValue()+"\n");
                }    
                for(int i=0;i<test.numInstances();i++){
                    temp=test.instance(i);
                    x= temp.relationalValue(0);
                    for(Instance y:x){//Each dimension
                        for(int j=0;j<y.numAttributes();j++)
                            uniTest.writeString(y.value(j)+",");
                    }
                    uniTest.writeString((int)temp.classValue()+"\n");
                }    

    //            System.out.println(" Object type ="+x);
                train = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TRAIN");
                System.out.println("Can load univariate "+dest+prob+"\\"+prob+"_TRAIN");
                test = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TEST");
                System.out.println("Can load univariate "+dest+prob+"\\"+prob+"_TEST");

            }
            else
                System.out.println("Already done "+prob);
        }
        
        
    }
    
    static enum MV_Classifiers {SHAPELETI, SHAPELETD, SHAPELET_INDEP, ED_I, ED_D, DTW_I, DTW_D, DTW_A}
    
    public static boolean isMultivariateClassifier(String classifier){
        for (MV_Classifiers mvClassifier: MV_Classifiers.values()){
            if (mvClassifier.name().toLowerCase().equals(classifier.toLowerCase())) {
                return true;
            }
        }
        return false;
    }
    
    //TODO CHECK TO SEE IF FILES ALREADY MADE
    public static Instances[] convertToUnivariate(String path, String dest, String prob){
        
        if (!CollateResults.validateSingleFoldFile(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN") 
                || !CollateResults.validateSingleFoldFile(dest+prob+"_UNI"+"/"+prob+"_UNI_TEST")){
        
            Instances train =ClassifierTools.loadData(path+prob+"/"+prob+"_TRAIN");
            Instances test =ClassifierTools.loadData(path+prob+"/"+prob+"_TEST");

            Instance temp=test.instance(0);
            Instances x= temp.relationalValue(0);

            int numAtts=x.numInstances()*x.numAttributes();

            File f= new File(dest+prob+"_UNI");
            f.mkdirs();
            OutFile uniTrain=new OutFile(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN.arff");
            OutFile uniTest=new OutFile(dest+prob+"_UNI"+"/"+prob+"_UNI_TEST.arff");
            String header="@relation "+prob+"\n";
            for(int i=0;i<numAtts;i++){
                header+="@attribute att"+i+" numeric \n";
            }
            header+="@attribute "+train.classAttribute().name()+ " {";
            for(int i=0;i<train.numClasses()-1;i++)
                header+=train.classAttribute().value(i)+",";
            header+=train.classAttribute().value(train.numClasses()-1)+"}\n";
            header+="@data \n";
            uniTrain.writeString(header);
            uniTest.writeString(header);
            for(int i=0;i<train.numInstances();i++){
                temp=train.instance(i);
                x= temp.relationalValue(0);
                for(Instance y:x){//Each dimension
                    for(int j=0;j<y.numAttributes();j++)
                        uniTrain.writeString(y.value(j)+",");
                }
                uniTrain.writeString(temp.classAttribute().value((int)temp.classValue())+"\n");
            }    
            for(int i=0;i<test.numInstances();i++){
                temp=test.instance(i);
                x= temp.relationalValue(0);
                for(Instance y:x){//Each dimension
                    for(int j=0;j<y.numAttributes();j++)
                        uniTest.writeString(y.value(j)+",");
                }
                if (temp.classIsMissing()){
                    uniTest.writeString("?\n");
                }
                else {
                    uniTest.writeString(temp.classAttribute().value((int)temp.classValue())+"\n");
                }
            } 
        }

    //            System.out.println(" Object type ="+x);
        Instances train = ClassifierTools.loadData(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");
        System.out.println("Can load univariate "+dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");
        Instances test = ClassifierTools.loadData(dest+prob+"_UNI"+"/"+prob+"_UNI_TEST");
        System.out.println("Can load univariate "+dest+prob+"_UNI"+"/"+prob+"_UNI_TEST");
        
        Instances[] i = new Instances[2];
        i[0] = train;
        i[1] = test;
        return i;
    }
    
    //TODO CHECK TO SEE IF FILES ALREADY MADE
    public static Instances convertToUnivariateTrain(String path, String dest, String prob){
        
        if (!CollateResults.validateSingleFoldFile(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN")){
        
            Instances train =ClassifierTools.loadData(path+prob+"/"+prob+"_TRAIN");

            Instance temp=train.instance(0);
            Instances x= temp.relationalValue(0);

            int numAtts=x.numInstances()*x.numAttributes();

            File f= new File(dest+prob+"_UNI");
            f.mkdirs();
            OutFile uniTrain=new OutFile(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN.arff");
            String header="@relation "+prob+"\n";
            for(int i=0;i<numAtts;i++){
                header+="@attribute att"+i+" numeric \n";
            }
            header+="@attribute "+train.classAttribute().name()+ " {";
            for(int i=0;i<train.numClasses()-1;i++)
                header+=train.classAttribute().value(i)+",";
            header+=train.classAttribute().value(train.numClasses()-1)+"}\n";
            header+="@data \n";
            uniTrain.writeString(header);
            for(int i=0;i<train.numInstances();i++){
                temp=train.instance(i);
                x= temp.relationalValue(0);
                for(Instance y:x){//Each dimension
                    for(int j=0;j<y.numAttributes();j++)
                        uniTrain.writeString(y.value(j)+",");
                }
                uniTrain.writeString(temp.classAttribute().value((int)temp.classValue())+"\n");
            }    
        }

    //            System.out.println(" Object type ="+x);
        Instances train = ClassifierTools.loadData(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");
        System.out.println("Can load univariate "+dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");

        return train;
    }

    public static void checkConcatenatedFiles(){
        String dest="Z:\\Data\\ConcatenatedMTSC\\";
        for(String prob:DataSets.mtscProblems2018){
               
//            System.out.println(" Object type ="+x);
            try{
                Instances train = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TRAIN");
            System.out.println("Can load univariate "+dest+prob+"\\"+prob+"_TRAIN");
            }catch(Exception e){
                System.out.println("UNABLE TO  LOAD :"+prob+" TRAIN FILE: EXCEPTION "+e);   
            }
            
            try{
                Instances test = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TEST");
            System.out.println("Can load univariate "+dest+prob+"\\"+prob+"_TEST");
            }catch(Exception e){
                System.out.println("UNABLE TO LOAD :"+prob+" TEST FILE: EXCEPTION "+e);   
            }
        }
        
        
    }

    public static void formatPhilData(){
        Instances multi=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\FinalMulti");
        Instances trans=MultivariateInstanceTools.transposeRelationalData(multi);
//       double[][] rawData=
        
        
//        Instances temp=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\FinalUni");
//        System.out.println(" Uni: num cases "+temp.numInstances()+" num atts ="+temp.numAttributes());
//        Instances mtsc=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,30);
        OutFile out=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports.arff");
        out.writeString(trans.toString());
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports.arff");
        System.out.println("New data = "+test);
        Instances[] split=InstanceTools.resampleInstances(test, 0, 0.5);
        OutFile train=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports\\RacketSports_TRAIN.arff");
        train.writeString(split[0].toString());
        OutFile testF=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports\\RacketSports_TEST.arff");
        testF.writeString(split[1].toString());
    }
    public static void splitData(String path,String prob){
        Instances all=ClassifierTools.loadData(path+prob+"\\"+prob);
        Instances[] split=InstanceTools.resampleInstances(all, 0, 0.5);
        OutFile out=new OutFile(path+prob+"\\"+prob+"_TRAIN.arff");
        out.writeLine(split[0].toString());
         out=new OutFile(path+prob+"\\"+prob+"_TEST.arff");
        out.writeLine(split[1].toString());
    }
    
    public static void formatDuckDuckGeese(){
        String path="Z:\\Data\\MultivariateTSCProblems\\DuckDuckGeese\\";
        Instances data=ClassifierTools.loadData(path+"DuckDuckGeese");
                Instance temp=data.instance(0);
                Instances x= temp.relationalValue(0);
                System.out.println("train number of dimensions "+x.numInstances());
                System.out.println("train number of attributes per dimension "+x.numAttributes());

        
        Instances[] split= MultivariateInstanceTools.resampleMultivariateInstances(data, 0, 0.6);
        System.out.println("Train size ="+split[0].numInstances());
        System.out.println("Test size ="+split[1].numInstances());
        OutFile out=new OutFile(path+"DuckDuckGeese_TRAIN.arff");
        out.writeString(split[0]+"");
        out=new OutFile(path+"DuckDuckGeese_TEST.arff");
        out.writeString(split[1]+"");
        
        
    }  
    
    
    public static void formatCricket(){
        String path="Z:\\Data\\Multivariate Working Area\\Cricket\\";
        Instances[] data=new Instances[6];
        data[0]=ClassifierTools.loadData(path+"CricketXLeft.arff");
        data[1]=ClassifierTools.loadData(path+"CricketYLeft.arff");
        data[2]=ClassifierTools.loadData(path+"CricketZLeft.arff");
        data[3]=ClassifierTools.loadData(path+"CricketXRight.arff");
        data[4]=ClassifierTools.loadData(path+"CricketYRight.arff");
        data[5]=ClassifierTools.loadData(path+"CricketZRight.arff");
        Instances all=MultivariateInstanceTools.mergeToMultivariateInstances(data);
        OutFile out=new OutFile(path+"Cricket.arff");
        System.out.println("Cricket number of instances ="+all.numInstances());
            Instance temp=all.instance(0);
            Instances x= temp.relationalValue(0);
            System.out.println(" number of dimensions "+x.numInstances());
            System.out.println(" number of attributes per dimension "+x.numAttributes());
        out.writeString(all+"");
        Instances[] split= MultivariateInstanceTools.resampleMultivariateInstances(all, 0, 0.6);
        System.out.println("Train size ="+split[0].numInstances());
        System.out.println("Test size ="+split[1].numInstances());
         out=new OutFile(path+"Cricket_TRAIN.arff");
        out.writeString(split[0]+"");
         out=new OutFile(path+"Cricket_TEST.arff");
        out.writeString(split[1]+"");

    
    }
    public static void makeSingleDimensionFiles(){
         String path="Z:\\Data\\MultivariateTSCProblems\\";
        for(String prob: DataSets.mtscProblems2018){
            
            File f= new File(path+prob+"\\"+prob+"Dimension"+(1)+"_TRAIN.arff");
            if(!f.exists()){
                Instances train =ClassifierTools.loadData(path+prob+"\\"+prob+"_TRAIN");
                Instances test =ClassifierTools.loadData(path+prob+"\\"+prob+"_TEST");
                System.out.println("PROBLEM "+prob);        
                System.out.println("Num train instances ="+train.numInstances());
                System.out.println("Num test instances ="+test.numInstances());
                System.out.println("num attributes (should be 2!)="+train.numAttributes());
                System.out.println("num classes="+train.numClasses());
                Instance temp=train.instance(0);
                Instances x= temp.relationalValue(0);
                System.out.println(" number of dimensions "+x.numInstances());
                System.out.println(" number of attributes per dimension "+x.numAttributes());
                Instances[] splitTest=MultivariateInstanceTools.splitMultivariateInstances(test);
                Instances[] splitTrain=MultivariateInstanceTools.splitMultivariateInstances(train);
                System.out.println(" Num split files ="+splitTest.length);
                for(int i=0;i<splitTrain.length;i++){
                    System.out.println("Number of test instances = "+splitTest[i].numInstances());
                    OutFile outTrain=new OutFile(path+prob+"\\"+prob+"Dimension"+(i+1)+"_TRAIN.arff");
                    outTrain.writeLine(splitTrain[i].toString()+"");
                    OutFile outTest=new OutFile(path+prob+"\\"+prob+"Dimension"+(i+1)+"_TEST.arff");
                    outTest.writeLine(splitTest[i].toString()+"");

                }
            }
            
//            System.out.println(" Object type ="+x);

        }   
    }
    
    public static void summariseData(){
        String path="Z:\\Data\\MultivariateTSCProblems\\";
        OutFile out=new OutFile("Z:\\Data\\MultivariateTSCProblems\\SummaryData.csv");
        out.writeLine("problem,numTrainCases,numTestCases,numDimensions,seriesLength,numClasses");
        for(String prob: DataSets.mtscProblems2018){
            Instances train =ClassifierTools.loadData(path+prob+"\\"+prob+"_TRAIN");
            Instances test =ClassifierTools.loadData(path+prob+"\\"+prob+"_TEST");
            System.out.println("PROBLEM "+prob);        
            System.out.println("Num train instances ="+train.numInstances());
            System.out.println("Num test instances ="+test.numInstances());
            System.out.println("num attributes (should be 2!)="+train.numAttributes());
            System.out.println("num classes="+train.numClasses());
            Instance temp=train.instance(0);
            Instances x= temp.relationalValue(0);
            System.out.println(" number of dimensions "+x.numInstances());
            System.out.println(" number of attributes per dimension "+x.numAttributes());
            out.writeLine(prob+","+train.numInstances()+","+test.numInstances()+","+x.numInstances()+","+x.numAttributes()+","+train.numClasses());

//            System.out.println(" Object type ="+x);

        }
        
        
        
    }
//1. Format into a standard flat ARFF, then make into a multivariate problem.  BCI II data set ia  
  public static void formatSelfRegulationSCP1() throws Exception {
      
    String path="C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 2\\Data Set 1a\\";
    InFile class1=new InFile(path+"Traindata_0.txt");
    InFile class2=new InFile(path+"Traindata_1.txt");
    OutFile arff=new OutFile(path+"SelfRegulationSCPUni_TRAIN.arff");
    
    int numC1=135;
    int numC2=133;
    int d=6;
        int m=896;
        
    
    arff.writeLine("@relation SelfRegulationSCP1");
    for(int i=1;i<=d*m;i++)
        arff.writeLine("@attribute att"+i+" real");
    arff.writeLine("@attribute cortical {negativity,positivity}");
   arff.writeLine("@data");
     
      for(int i=0;i<numC1;i++){
          String line=class1.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arff.writeString(split[j]+",");
          arff.writeLine("negativity");
          
      }

      for(int i=0;i<numC2;i++){
          String line=class2.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arff.writeString(split[j]+",");
          arff.writeLine("positivity");
      }
      arff.closeFile();
      Instances temp=ClassifierTools.loadData(path+"SelfRegulationSCP1Uni_TRAIN.arff");
      Instances multi=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,896);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arff=new OutFile(path+"SelfRegulationSCP1_TRAIN.arff");
      arff.writeLine(multi.toString());
      
        int testSize=293;   
     InFile test=new InFile(path+"TestData.txt");
    arff=new OutFile(path+"SelfRegulationSCP1Uni_TEST.arff");
    arff.writeLine("@relation SelfRegulationSCP1");
    for(int i=1;i<=d*m;i++)
        arff.writeLine("@attribute att"+i+" real");
    arff.writeLine("@attribute cortical {negativity,positivity}");
   arff.writeLine("@data");
     
      for(int i=0;i<testSize;i++){
          String line=test.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arff.writeString(split[j]+",");
          if(split[0].equals("0.00"))
              arff.writeLine("negativity");
          else
              arff.writeLine("positivity");
      }
       temp=ClassifierTools.loadData(path+"SelfRegulationSCPUni_TEST.arff");
      multi=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,896);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arff=new OutFile(path+"SelfRegulationSCP1_TEST.arff");
      arff.writeLine(multi.toString());
      
    
        
        
  }    
    

//1. Format into a standard flat ARFF, then make into a multivariate problem.  BCI II data set ib  
  public static void formatSelfRegulationSCP2() throws Exception {
      
    String path="C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 2\\Data Set 1b\\";
    InFile class1=new InFile(path+"Traindata_0.txt");
    InFile class2=new InFile(path+"Traindata_1.txt");
    OutFile arff=new OutFile(path+"SelfRegulationSCP2Uni_TRAIN.arff");
    
    int numC1=100;
    int numC2=100;
    int d=7;
        int m=1152;
        
    
    arff.writeLine("@relation SelfRegulationSCP2");
    for(int i=1;i<=d*m;i++)
        arff.writeLine("@attribute att"+i+" real");
    arff.writeLine("@attribute cortical {negativity,positivity}");
   arff.writeLine("@data");
     
      for(int i=0;i<numC1;i++){
          String line=class1.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arff.writeString(split[j]+",");
          arff.writeLine("negativity");
          
      }

      for(int i=0;i<numC2;i++){
          String line=class2.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arff.writeString(split[j]+",");
          arff.writeLine("positivity");
      }
      arff.closeFile();
      Instances temp=ClassifierTools.loadData(path+"SelfRegulationSCP2Uni_TRAIN.arff");
      Instances multi=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,m);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arff=new OutFile(path+"SelfRegulationSCP2_TRAIN.arff");
      arff.writeLine(multi.toString());
      
        int testSize=180;   
     InFile test=new InFile(path+"TestData.txt");
    arff=new OutFile(path+"SelfRegulationSCP2Uni_TEST.arff");
    arff.writeLine("@relation SelfRegulationSCP2");
    for(int i=1;i<=d*m;i++)
        arff.writeLine("@attribute att"+i+" real");
    arff.writeLine("@attribute cortical {negativity,positivity}");
   arff.writeLine("@data");
     
      for(int i=0;i<testSize;i++){
          String line=test.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arff.writeString(split[j]+",");
          if(split[0].equals("0.00"))
              arff.writeLine("negativity");
          else
              arff.writeLine("positivity");
      }
       temp=ClassifierTools.loadData(path+"SelfRegulationSCP2Uni_TEST.arff");
      multi=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,m);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arff=new OutFile(path+"SelfRegulationSCP2_TEST.arff");
      arff.writeLine(multi.toString());
      
    
        
        
  }    
    

//1. Format into a standard flat ARFF, then make into a multivariate problem.  BCI II data set IV 
  public static void formatFingerMovements() throws Exception {
      
    String path="C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 2\\Data Set IV\\";
    InFile train=new InFile(path+"sp1s_aa_train.txt");
    OutFile arffTrain=new OutFile(path+"FingerMovementsUni_TRAIN.arff");
    int d=28;
    int m=50;
    int trainSize=316;  
    int testSize=100;   
    
    arffTrain.writeLine("@relation FingerMovements");
    for(int i=1;i<=d*m;i++)
        arffTrain.writeLine("@attribute att"+i+" real");
    arffTrain.writeLine("@attribute hand {left,right}");
    arffTrain.writeLine("@data");
     
      for(int i=0;i<trainSize;i++){
          String line=train.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arffTrain.writeString(split[j]+",");
          if(split[0].equals("0.00"))
              arffTrain.writeLine("left");
          else
              arffTrain.writeLine("right");
      }
      Instances temp=ClassifierTools.loadData(path+"FingerMovementsUni_TRAIN.arff");
      Instances multi=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,m);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arffTrain=new OutFile(path+"FingerMovements_TRAIN.arff");
      arffTrain.writeLine(multi.toString());
      
          InFile test=new InFile(path+"sp1s_aa_test.txt");
    OutFile arffTest=new OutFile(path+"FingerMovementsUni_TEST.arff");
      
     arffTest.writeLine("@relation FingerMovements");
    for(int i=1;i<=d*m;i++)
        arffTest.writeLine("@attribute att"+i+" real");
    arffTest.writeLine("@attribute hand {left,right}");
    arffTest.writeLine("@data");
     
      for(int i=0;i<testSize;i++){
          String line=test.readLine();
          String[] split=line.split("\\s+");
          for(int j=1;j<=d*m;j++)
              arffTest.writeString(split[j]+",");
          if(split[0].equals("0.00"))
              arffTest.writeLine("left");
          else
              arffTest.writeLine("right");
      }
      temp=ClassifierTools.loadData(path+"FingerMovementsUni_TEST.arff");
      multi=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,m);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arffTrain=new OutFile(path+"FingerMovements_TEST.arff");
      arffTrain.writeLine(multi.toString());
    
        
        
  }    
    
  
  public static void formatCharacterTrajectories() throws Exception {
//#classes=  20, d=3, length=109-205, train 6600, test 2200

        
        InFile train = new InFile("");
        InFile test = new InFile("");
        OutFile trainarff = new OutFile("");
        OutFile testarff = new OutFile("");
        String line=train.readLine();
        while(line!=null){
//            String[] split
            
        }
        
        
    }
  //BCI 3 Dataset 1
    public static void formatMotorImagery(){
//Each channel is on a different line in the text file.   
//Labels in a separate text file
        int m=3000;
        int d=64;
        int trainSize=278;
        int testSize=100;
        InFile trainCSV=new InFile("C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\Competition_train_cnt.csv");
        InFile testCSV=new InFile("C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\Competition_test_cnt.csv");
        InFile trainLabels=new InFile("C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\Competition_train_lab.txt");
        InFile testLabels=new InFile("C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\Test Set Labels.txt");
        String arffP="C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\MotorImageryUni_TRAIN.arff";
        String arffP2="C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\MotorImageryUni_TEST.arff";
        OutFile arffTrain=new OutFile(arffP);
        arffTrain.writeLine("@relation MotorImagery");
        for(int i=1;i<=d*m;i++)
            arffTrain.writeLine("@attribute att"+i+" real");
        arffTrain.writeLine("@attribute motion{finger,tongue}");
        arffTrain.writeLine("@data");
        for(int i=0;i<trainSize;i++){
            for(int j=0;j<d;j++)
                arffTrain.writeString(trainCSV.readLine()+",");
            int label=trainLabels.readInt();
            if(label==-1)
                arffTrain.writeLine("finger");
            else
                arffTrain.writeLine("tongue");
        }
        arffTrain.closeFile();
        Instances tr=ClassifierTools.loadData(arffP);
        System.out.println("Num instances ="+tr.numInstances()+" num atts ="+tr.numAttributes());
       Instances multi=MultivariateInstanceTools.convertUnivariateToMultivariate(tr,m);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arffTrain=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\MotorImagery_TRAIN.arff");
      arffTrain.writeLine(multi.toString());
       
         OutFile arffTest=new OutFile(arffP2);
        arffTest.writeLine("@relation MotorImagery");
        for(int i=1;i<=d*m;i++)
            arffTest.writeLine("@attribute att"+i+" real");
        arffTest.writeLine("@attribute motion{finger,tongue}");
        arffTest.writeLine("@data");
        for(int i=0;i<testSize;i++){
            for(int j=0;j<d;j++)
                arffTest.writeString(testCSV.readLine()+",");
            int label=testLabels.readInt();
            if(label==-1)
                arffTest.writeLine("finger");
            else
                arffTest.writeLine("tongue");
        }
        arffTest.closeFile();
        Instances te=ClassifierTools.loadData(arffP2);
        System.out.println("Num instances ="+te.numInstances()+" num atts ="+te.numAttributes());
        multi=MultivariateInstanceTools.convertUnivariateToMultivariate(te,m);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arffTest=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\MotorImagery_TEST.arff");
      arffTest.writeLine(multi.toString());
        
        
        System.out.println("TEST Num instances ="+te.numInstances()+" num atts ="+te.numAttributes());
       
    }
    public static void main(String[] args) throws Exception {
        
        formatDuckDuckGeese();        
//        formatCricket();
         System.exit(0);
        makeSingleDimensionFiles();
         summariseData();
        debugFormat();
        makeConcatenatedFiles();
        String prob="UWaveGestureLibrary";
        String dest="Z:\\Data\\UnivariateMTSC\\";
        String path="Z:\\Data\\Multivariate TSC Problems\\";
        Instances test = ClassifierTools.loadData(path+prob+"\\"+prob+"_TEST");
    Instances train = ClassifierTools.loadData(path+prob+"\\"+prob+"_TRAIN");
        
       Instances test2 = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TEST");
  Instances train2 = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TRAIN");
        
//        summariseData();
//        System.exit(0);
//        debugFormat();
//        makeUnivariateFiles();
//        String prob="UWaveGestureLibrary";
//        String dest="Z:\\Data\\UnivariateMTSC\\";
//        String path="Z:\\Data\\Multivariate TSC Problems\\";
//        Instances test = ClassifierTools.loadData(path+prob+"\\"+prob+"_TEST");
//    Instances train = ClassifierTools.loadData(path+prob+"\\"+prob+"_TRAIN");
//        
//       Instances test2 = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TEST");
//  Instances train2 = ClassifierTools.loadData(dest+prob+"\\"+prob+"_TRAIN");
//
//        
// //       checkUnivariateFiles();
////        formatMotorImagery();
// //       formatFingerMovements();
//        //formatSelfRegulationSCP1();
// //       formatSelfRegulationSCP2();
//        //        formatPhilData();
// //       splitData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Multivariate Working Area\\Michael_Unfinalised\\","Phoneme");
//     //  summariseData();        
//
//        //gettingStarted();
//       // mergeEpilepsy();

        //Instances[] data = convertToUnivariate("C:/UEAMachineLearning/Datasets/Kaggle/PLAsTiCCAstronomicalClassification/", "C:/UEAMachineLearning/Datasets/Kaggle/PLAsTiCCAstronomicalClassification/", "LSSTTrain");
        //System.out.println(data[0]);
        //System.out.println(data[1]);
    }
    public static void debugFormat(){
//        ECGActivities
        Instances train,test;
        train=ClassifierTools.loadData("Z:\\Data\\MultivariateTSCProblems\\ECGActivities\\ECGActivities_TRAIN");
        test=ClassifierTools.loadData("Z:\\Data\\MultivariateTSCProblems\\ECGActivities\\ECGActivities_TEST");
        
//        Instances[] split=InstanceTools.resampleTrainAndTestInstances(train, test, 1);
        Instances[] split=MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances(train, test, 1);
        System.out.println("IS it relational ? "+split[0].checkForAttributeType(Attribute.RELATIONAL));
        System.out.println("IS it relational ? "+split[0].checkForAttributeType(Attribute.RELATIONAL));

        System.out.println("Fold 1 TRAIN num instances "+split[0].numInstances()+" Num atts ="+(split[0].numAttributes()-1));
//        System.out.println(split[0]+"");
        System.out.println("Fold 1 TRAIN  instance 1 num dimensions "+split[0].instance(0).relationalValue(0).numInstances()+" series length "+split[0].instance(0).relationalValue(0).numAttributes());
        for(Instance ins:split[0])
        System.out.println("Fold TRAIN  instance num dimensions "+ins.relationalValue(0).numInstances()+" series length "+ins.relationalValue(0).numAttributes());

    }
    public static void mergeEpilepsy(){
        Instances x,y,z;
        Instances all;
        String sourcePath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\EpilepsyX\\";
        String destPath="C:\\Users\\ajb\\Dropbox\\Multivariate TSC Problems\\HAR\\Epilepsy\\";
        x=ClassifierTools.loadData(sourcePath+"EpilepsyX_ALL");
        y=ClassifierTools.loadData(sourcePath+"EpilepsyY_ALL");
        z=ClassifierTools.loadData(sourcePath+"EpilepsyZ_ALL");
//Delete the use ID, will reinsert manually after        
        x.deleteAttributeAt(0);
        y.deleteAttributeAt(0);
        z.deleteAttributeAt(0);
        all=utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(new Instances[]{x,y,z});
//        OutFile out=new OutFile(destPath+"EpilepsyNew.arff");
//        out.writeString(all.toString());
 //Create train test splits so participant 1,2,3 in train and 4,5,6 in test       
        int trainSize=149;
        int testSize=126;
        Instances train= new Instances(all,0);
        Instances test= new Instances(all);
        for(int i=0;i<trainSize;i++){
            Instance t= test.remove(0);
            train.add(t);
        }
        OutFile tr=new OutFile(destPath+"Epilepsy_TRAIN.arff");
        OutFile te=new OutFile(destPath+"Epilepsy_TEST.arff");
        tr.writeString(train.toString());
        te.writeString(test.toString());
        
        
    }
    
/**A getting started with relational attributes in Weka. Once you have the basics
 * there are a range of tools for manipulating them in 
 * package utilities.multivariate_tools 
 * 
 * See https://weka.wikispaces.com/Multi-instance+classification
 * for more     
 * */
    public static void gettingStarted(){
//Load a multivariate data set
        String path="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Multivariate\\univariateConcatExample";
        Instances train = ClassifierTools.loadData(path);
        System.out.println(" univariate data = "+train);
        path="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Multivariate\\multivariateConcatExample";
        train = ClassifierTools.loadData(path);
        System.out.println(" multivariate data = "+train);
//Recover the first instance
        Instance first=train.instance(0);
//Split into separate dimensions
        Instances split=first.relationalValue(0);
        System.out.println(" A single multivariate case split into 3 instances with no class values= "+split);
        for(Instance ins:split)
            System.out.println("Dimension of first case =" +ins);
//Extract as arrays
        double[][] d = new double[split.numInstances()][];
        for(int i=0;i<split.numInstances();i++)
           d[i]=split.instance(i).toDoubleArray();

    
    }
}
