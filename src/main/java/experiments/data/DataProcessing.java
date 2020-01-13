/*
Multivariate data can be stored in Wekas "multi instance" format
https://weka.wikispaces.com/Multi-instance+classification

for TSC, the basic univariate syntax is 

 */
package experiments.data;

import experiments.CollateResults;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import tsml.classifiers.multivariate.NN_ED_I;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Sorting out the new archive, some general utility functions
 * @author ajb
 */
public class DataProcessing {


    public static void collateShapeletParameters(String readPath,String classifier,String[] problems) throws Exception {
//ONE FOLD ONLY
//1. List of full transforms vs random
        int count=0;
        int full=0;
        int withinContract=0;
        File of=new File(readPath+classifier+"/ParameterSummary/");
        of.mkdirs();

        OutFile timings = new OutFile(readPath+classifier+"/ParameterSummary/BuildTime"+classifier+".csv");
        OutFile numShapelets = new OutFile(readPath+classifier+"/ParameterSummary/NumShapelets"+classifier+".csv");
        OutFile combo = new OutFile(readPath+classifier+"/ParameterSummary/combo"+classifier+".csv");
        OutFile timeRegression = new OutFile(readPath+classifier+"/ParameterSummary/singleShapeletTrainTime.arff");
        timeRegression.writeLine("@Relation ShapeletTrainTimeRegression");
        timeRegression.writeLine("@Attribute dataSet String");
        timeRegression.writeLine("@Attribute log(n) real");
        timeRegression.writeLine("@Attribute log(m) real");
        timeRegression.writeLine("@Attribute log(s) real");
        timeRegression.writeLine("@data");

        //FORMAT by column: SearchType (full/random), transformContract (secs), transformActual (secs), proportion
        DecimalFormat df= new DecimalFormat("##.##");
        double meanProportion=0;
        double meanOutForFull=0;
        timings.writeLine("problem,SearchType(Full/Random), transformContract(secs),transformActual(secs),proportionTimeUsed,classifierTime");
        numShapelets.writeLine("problem,SearchType(Full/Random),numShapeletsInProblem,ProportionToEvaluate," +
                "numShapeletsInTransform,NumberShapeletsEvaluated,NumberShapeletsEarlyAbandoned");
        combo.writeString("problem,SearchType(Full/Random), transformContract(secs),transformActual(secs),proportionTimeUsed,classifierTime");
        combo.writeLine(",numShapeletsInProblem,ProportionToEvaluate,NumToEvaluate," +
                "NumInTransform,NumberEvaluated,NumberEarlyAbandoned,TotalNumber,TimePerShapelet,FullTimeEstimate(hrs),withinContract");
        for (String problem : problems) {
//            Instances data=DatasetLoading.loadData(""+problem+"/"+problem+"_TRAIN.arff");
            for(int i=0;i<30;i++) {
                File f = new File(readPath + classifier + "/Predictions/" + problem + "/testFold"+i+".csv");
                if (f.exists()) {
                    timings.writeString(problem);
                    numShapelets.writeString(problem);
                    combo.writeString(problem);
                    count++;
                    InFile inf = new InFile(readPath + classifier + "/Predictions/" + problem + "/testFold"+i+".csv");
                    String str = inf.readLine();
                    str = inf.readLine();
                    String[] split = str.split(",");
                    System.out.println(problem + "  Full/Random = " + split[22]);
                    timings.writeString("," + split[11]); //Full/Random
                    combo.writeString("," + split[11]); //Full/Random
                    double contract = Double.parseDouble(split[5]);//Contracted time
                    contract /= 1000000000.0;
                    timings.writeString("," + df.format(contract));//Contract time
                    combo.writeString("," + df.format(contract));
                    double actual = Double.parseDouble(split[1]); //Actual transform time
                    actual /= 1000000000.0;

                    timings.writeString("," + df.format(actual)); //Actual time
                    combo.writeString("," + df.format(actual));
                    if (contract > 0) {
                        timings.writeString("," + df.format(actual / contract));//Proportion
                        combo.writeString("," + df.format(actual / contract));
                    } else {
                        timings.writeString("," + 1.0);//Proportion
                        combo.writeString("," + 1.0);
                    }

                    String str2 = inf.readLine();
                    String[] split2 = str2.split(",");
                    double totalTime = Double.parseDouble(split[1]);//Total Time

                    totalTime /= 1000000000.0;
                    timings.writeLine("," + df.format(totalTime - actual));//Classifier time
                    combo.writeString("," + df.format(totalTime - actual));

                    if (split[11].equals("FULL")) {
                        full++;
                        if (actual < contract)
                            withinContract++;
                        else
                            meanOutForFull += actual / contract;
                    } else
                        meanProportion += actual / contract;
                    numShapelets.writeString("," + split[11]); //Full/Random
                    numShapelets.writeString("," + split[7]);//Shapelets in problem
                    int totalShapelets = Integer.parseInt(split[7]);
                    numShapelets.writeString("," + split[9]);//Proportion to evaluate
                    numShapelets.writeString("," + split[33]);//Shapelets in transform
                    numShapelets.writeString("," + split[35]);//Shapelets fully evaluated
                    numShapelets.writeLine("," + split[37]);//Shapelets abandoned
                    combo.writeString("," + split[7]);//Shapelets in problem
                    combo.writeString("," + split[9]);//Proportion to evaluate
                    int toEvaluate = (int) (Double.parseDouble(split[7]) * Double.parseDouble(split[9]));
                    combo.writeString("," + toEvaluate);//Number to evaluate for full
                    combo.writeString("," + split[33]);//Shapelets in transform
                    combo.writeString("," + split[35]);//Shapelets evaluated
                    combo.writeString("," + split[37]);//Shapelets abandoned
                    double totalEvals = Double.parseDouble(split[35]) + Double.parseDouble(split[37]);
                    System.out.println("Total evals");
                    combo.writeString("," + (long) totalEvals);//number actually evaluated
                    double timePerS = actual / totalEvals;
                    combo.writeString("," + timePerS);//Time per shapelet (secs)
                    double hrsToFull = timePerS * totalShapelets / (60 * 60);
                    combo.writeString("," + hrsToFull);//Time for full
                    if (timePerS * totalShapelets < contract || split[11].equals("FULL"))
                        combo.writeLine(",YES");
                    else
                        combo.writeLine(",NO");
                    timeRegression.writeLine(timePerS+"");

                }
            }
        }
        timings.closeFile();
        numShapelets.closeFile();
        System.out.println(count+" problems present");
        System.out.println(" Mean proportion actual/contract for random ="+meanProportion/(count-full));
        System.out.println(" number of full transforms = "+full+" number full actually within contract ="+withinContract);
        System.out.println(" Mean proportion fill given out of contract "+meanOutForFull/(full-withinContract));
    }



    public static void makeZips(String[] directories, String dest,String ... source) {
        File inf=new File(dest);
        inf.mkdirs();
        for(String str: directories){
            // create byte buffer
            byte[] buffer = new byte[1024];
            FileOutputStream fos;
            ZipOutputStream zos =null;            
            try {
                fos = new FileOutputStream(dest+str+".zip");
                zos= new ZipOutputStream(fos);
                
                for(String src:source){
                    File dir = new File(src+str);
                    File[] files = dir.listFiles();
                    for (int i = 0; i < files.length; i++) {
                        System.out.println("Adding file: " + files[i].getName());
                        if(files[i].isFile()){
                            FileInputStream fis = new FileInputStream(files[i]);
                            // begin writing a new ZIP entry, positions the stream to the start of the entry data

                            zos.putNextEntry(new ZipEntry(files[i].getName()));
                            int length;
                            while ((length = fis.read(buffer)) > 0) {
                                zos.write(buffer, 0, length);
                            }
                            zos.closeEntry();
                            // close the InputStream
                            fis.close();
                        }
                    }
                }
                zos.close();
            } catch (FileNotFoundException ex) {
                System.out.println("ERROR OPENING THE ZIP on "+dest+str+".zip");
            } catch (IOException ex) {
                System.out.println("ERROR CLOSING THE ZIP on "+dest+str+".zip"+" Exception ="+ex);
            }
        } 
        

    }    
    public static void concatenateShapelets(){
       String path="E:\\Data\\ShapeletTransforms\\";
       String[] timeLength={"1"};
       String st = "ShapelelTransform";
       String hybrid="Hybrid";
       String combo="Combo";
       for(String s:timeLength){
           File f= new File(path+combo+s);
           f.mkdirs();
           for(String str:DatasetLists.tscProblems2018){
//Check they are present
               File tr1,tr2,te1,te2;
               tr1=new File(path+st+s+"\\Transforms\\"+str+"\\"+str+"_TRAIN.arff");
               te1=new File(path+st+s+"\\Transforms\\"+str+"\\"+str+"_TEST.arff");
               tr2=new File(path+st+s+"\\Transforms\\"+str+"\\"+str+"_TRAIN.arff");
               te2=new File(path+st+s+"\\Transforms\\"+str+"\\"+str+"_TEST.arff");
               Instances train1,train2;
                Instances test1,test2;
           }
//Load the data

//Check class labels are alligned

//Merge the instances

//Write to new files
           
           
       }
    
    }
    public static void makeAllZipFiles(){
       String[] paths={"Z:\\ArchiveData\\Univariate_arff\\","Z:\\ArchiveData\\Univariate_ts\\"};
       String dest="E:\\ArchiveData\\Zips_Univariate\\";
       String[] probs={"Adiac"};//DatasetLists.mtscProblems2018
       makeZips(probs,dest,paths);
//       makeZips(DatasetLists.mtscProblems2018,path,dest);
        
    }
    public static void checkZipFiles(String[] fileNames,String path, String dest){
       for(String str:fileNames){
           try {
               URL zip=new URL(path+str+".zip");
               long s=Files.copy(zip.openStream(),Paths.get(dest+str+".zip"),StandardCopyOption.REPLACE_EXISTING);
               System.out.println("Connected and opened "+path+str+".zip size = "+s);
           } catch (MalformedURLException ex) {
               System.out.println("UNABLE TO CONNECT TO ZIP FILE "+path+str+".zip");
               System.exit(0);
           } catch (IOException ex) {
               System.out.println("UNABLE TO OPEN AND COPY ZIP FILE "+path+str+".zip");
               System.exit(0);
           }
       }
        
        
    }

    public static void checkAllZipFiles(){
       String path="http://www.timeseriesclassification.com/Downloads/";
       String dest="C:\\temp\\Zips\\";
       File f=new File(dest);
       f.mkdirs();
//       checkZipFiles(DatasetLists.tscProblems2018,path,dest);
       checkZipFiles(DatasetLists.mtscProblems2018,path,dest);
        
    }
    
    
    public static void makeConcatenatedFiles(){
        String path="Z:\\Data\\Multivariate TSC Problems\\";
        String dest="Z:\\Data\\ConcatenatedMTSC\\";
        OutFile out=new OutFile(path+"SummaryData.csv");
        out.writeLine("problem,numTrainCases,numTestCases,numDimensions,seriesLength,numClasses");
        String[] probs={"BasicMotions"};
        for(String prob:DatasetLists.mtscProblems2018){
            File t1=new File(dest+prob+"\\"+prob+"_TRAIN.arff");
            File t2=new File(dest+prob+"\\"+prob+"_TRAIN.arff");
            if(!(t1.exists()||t2.exists())){
                Instances train =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
                Instances test =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
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
                train = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TRAIN");
                System.out.println("Can load univariate "+dest+prob+"\\"+prob+"_TRAIN");
                test = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TEST");
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
        
            Instances train =DatasetLoading.loadDataNullable(path+prob+"/"+prob+"_TRAIN");
            Instances test =DatasetLoading.loadDataNullable(path+prob+"/"+prob+"_TEST");

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
        Instances train = DatasetLoading.loadDataNullable(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");
        System.out.println("Can load univariate "+dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");
        Instances test = DatasetLoading.loadDataNullable(dest+prob+"_UNI"+"/"+prob+"_UNI_TEST");
        System.out.println("Can load univariate "+dest+prob+"_UNI"+"/"+prob+"_UNI_TEST");
        
        Instances[] i = new Instances[2];
        i[0] = train;
        i[1] = test;
        return i;
    }
    
    //TODO CHECK TO SEE IF FILES ALREADY MADE
    public static Instances convertToUnivariateTrain(String path, String dest, String prob){
        
        if (!CollateResults.validateSingleFoldFile(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN")){
        
            Instances train =DatasetLoading.loadDataNullable(path+prob+"/"+prob+"_TRAIN");

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
        Instances train = DatasetLoading.loadDataNullable(dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");
        System.out.println("Can load univariate "+dest+prob+"_UNI"+"/"+prob+"_UNI_TRAIN");

        return train;
    }

    public static void checkConcatenatedFiles(){
        String dest="Z:\\Data\\ConcatenatedMTSC\\";
        for(String prob:DatasetLists.mtscProblems2018){
               
//            System.out.println(" Object type ="+x);
            try{
                Instances train = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TRAIN");
            System.out.println("Can load univariate "+dest+prob+"\\"+prob+"_TRAIN");
            }catch(Exception e){
                System.out.println("UNABLE TO  LOAD :"+prob+" TRAIN FILE: EXCEPTION "+e);   
            }
            
            try{
                Instances test = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TEST");
            System.out.println("Can load univariate "+dest+prob+"\\"+prob+"_TEST");
            }catch(Exception e){
                System.out.println("UNABLE TO LOAD :"+prob+" TEST FILE: EXCEPTION "+e);   
            }
        }
        
        
    }

    public static void formatPhilData(){
        Instances multi=DatasetLoading.loadDataNullable("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\FinalMulti");
        Instances trans=MultivariateInstanceTools.transposeRelationalData(multi);
//       double[][] rawData=
        
        
//        Instances temp=DatasetLoading.loadDataNullable("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\FinalUni");
//        System.out.println(" Uni: num cases "+temp.numInstances()+" num atts ="+temp.numAttributes());
//        Instances mtsc=MultivariateInstanceTools.convertUnivariateToMultivariate(temp,30);
        OutFile out=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports.arff");
        out.writeString(trans.toString());
        Instances test=DatasetLoading.loadDataNullable("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports.arff");
        System.out.println("New data = "+test);
        Instances[] split=InstanceTools.resampleInstances(test, 0, 0.5);
        OutFile train=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports\\RacketSports_TRAIN.arff");
        train.writeString(split[0].toString());
        OutFile testF=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\Multivariate TSC Problems\\RacketSports\\RacketSports_TEST.arff");
        testF.writeString(split[1].toString());
    }
    public static void splitData(String path,String prob){
        Instances all=DatasetLoading.loadDataNullable(path+prob+"\\"+prob);
        Instances[] split=InstanceTools.resampleInstances(all, 0, 0.5);
        OutFile out=new OutFile(path+prob+"\\"+prob+"_TRAIN.arff");
        out.writeLine(split[0].toString());
         out=new OutFile(path+prob+"\\"+prob+"_TEST.arff");
        out.writeLine(split[1].toString());
    }
    
    public static void formatDuckDuckGeese(){
        String path="Z:\\Data\\MultivariateTSCProblems\\DuckDuckGeese\\";
        Instances data=DatasetLoading.loadDataNullable(path+"DuckDuckGeese");
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
        data[0]=DatasetLoading.loadDataNullable(path+"CricketXLeft.arff");
        data[1]=DatasetLoading.loadDataNullable(path+"CricketYLeft.arff");
        data[2]=DatasetLoading.loadDataNullable(path+"CricketZLeft.arff");
        data[3]=DatasetLoading.loadDataNullable(path+"CricketXRight.arff");
        data[4]=DatasetLoading.loadDataNullable(path+"CricketYRight.arff");
        data[5]=DatasetLoading.loadDataNullable(path+"CricketZRight.arff");
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
    public static void makeSingleDimensionFiles(String path, String[] probs,boolean overwrite){
        for(String prob: probs){
            System.out.println("Processing "+prob);
            if(prob.equals("InsectWingbeat")||prob.equals("FaceDetection")|| prob.equals("DuckDuckGeese"))
                continue;
            File f= new File(path+prob+"\\"+prob+"Dimension"+(1)+"_TRAIN.arff");

            if(f.exists()&&!overwrite)
                continue;
            Instances train =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
            Instances test =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
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
            
//            System.out.println(" Object type ="+x);

        }   
    }
    public static void checkSpeechMarks(){
        String path="Z:\\Data\\MultivariateTSCProblems\\";
        OutFile out=new OutFile("Z:\\Data\\MultivariateTSCProblems\\SummaryData.csv");
        out.writeLine("problem,numTrainCases,numTestCases,numDimensions,seriesLength,numClasses");
        for(String prob: DatasetLists.mtscProblems2018){
            InFile[] split = new InFile[2];
            split[0] =new InFile(path+prob+"\\"+prob+"_TRAIN.arff");
            split[1] =new InFile(path+prob+"\\"+prob+"_TEST.arff");
//Ignore header      
            for(InFile f:split){
                String line=f.readLine();
                while(!line.startsWith("@data"))
                    line=f.readLine();
                line=f.readLine();
                while(line!=null && !line.contains("\""))
                    line=f.readLine();
                if(line!=null){
                    System.out.println("Problem "+prob+" contains speech marks "+line);  
                }
            }
        }
    }

    public static void removeSpeechMarks(){
        String[] problems={"ERing","JapaneseVowels","PenDigits","SpokenArabicDigits"};
        String path="Z:\\Data\\MultivariateTSCProblems\\";
        String path2="Z:\\Data\\Temp\\";
        for(String prob: problems){
            InFile[] split = new InFile[2];
            split[0] =new InFile(path+prob+"\\"+prob+"_TRAIN.arff");
            split[1] =new InFile(path+prob+"\\"+prob+"_TEST.arff");
            OutFile[] split2 = new OutFile[2];
            File file= new File(path2+prob);
            file.mkdirs();
            split2[0] =new OutFile(path2+prob+"\\"+prob+"_TRAIN.arff");
            split2[1] =new OutFile(path2+prob+"\\"+prob+"_TEST.arff");
//Ignore header      
            for(int i=0;i<split.length;i++){
                String line=split[i].readLine();
                while(!line.startsWith("@data")){
                    split2[i].writeLine(line);
                    line=split[i].readLine();
                    
                }
                split2[i].writeLine(line);
                line=split[i].readLine();
                while(line!=null){
                    String replaceString=line.replace("\"","'");
                    split2[i].writeLine(replaceString);
                    line=split[i].readLine();
                }
                if(line!=null){
                    System.out.println("SHOULD NOT GET HERE!");  
                }
            }
        }
    }
    public static void summariseMultivariateData(){
        String path="Z:\\Data\\MultivariateTSCProblems\\";
        OutFile out=new OutFile("Z:\\Data\\MultivariateTSCProblems\\SummaryData.csv");
        out.writeLine("problem,numTrainCases,numTestCases,numDimensions,seriesLength,numClasses");
        for(int i=0;i<DatasetLists.mtscProblems2018.length;i++){
            String prob=DatasetLists.mtscProblems2018[i]; 
            System.out.println("PROBLEM "+prob);        
            Instances train =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
            Instances test =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
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

    public static void summariseUnivariateData(String path){
//        String path="Z:\\Data\\TSCProblems2018\\";
        OutFile out=new OutFile(path+"SummaryData.csv");
        out.writeLine("problem,numTrainCases,numTestCases,seriesLength,numClasses");
        for(int i=0;i<DatasetLists.tscProblems2018.length;i++){
            String prob=DatasetLists.tscProblems2018[i]; 
            try{
                System.out.println("PROBLEM "+prob);        
                Instances train =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
                Instances test =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
                System.out.println("Num train instances ="+train.numInstances());
                System.out.println("Num test instances ="+test.numInstances());
                System.out.println("num attributes ="+(train.numAttributes()-1));
                System.out.println("num classes="+train.numClasses());
                out.writeLine(prob+","+train.numInstances()+","+test.numInstances()+","+(train.numAttributes()-1)+","+train.numClasses());
            }catch(Exception e){
                System.out.println("ERROR loading file "+prob);
            }
//            System.out.println(" Object type ="+x);

        }
        
        
        
    }


    public static void testSimpleClassifier() throws Exception{
        String path="Z:\\Data\\MultivariateTSCProblems\\";
        for(int i=15;i<DatasetLists.mtscProblems2018.length;i++){
            String prob=DatasetLists.mtscProblems2018[i]; 
            System.out.println("PROBLEM "+prob);        
            Instances train =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
            Instances test =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
            System.out.println("Num train instances ="+train.numInstances());
            System.out.println("Num test instances ="+test.numInstances());
            System.out.println("num attributes (should be 2!)="+train.numAttributes());
            System.out.println("num classes="+train.numClasses());
            Instance temp=train.instance(0);
            Instances x= temp.relationalValue(0);
            System.out.println(" number of dimensions "+x.numInstances());
            System.out.println(" number of attributes per dimension "+x.numAttributes());
            NN_ED_I nb = new NN_ED_I();
            nb.buildClassifier(train);
            double a=ClassifierTools.accuracy(test, nb);
            System.out.println("Problem ="+prob+" 1-NN ED accuracy  ="+a);

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
      Instances temp=DatasetLoading.loadDataNullable(path+"SelfRegulationSCP1Uni_TRAIN.arff");
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
       temp=DatasetLoading.loadDataNullable(path+"SelfRegulationSCPUni_TEST.arff");
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
      Instances temp=DatasetLoading.loadDataNullable(path+"SelfRegulationSCP2Uni_TRAIN.arff");
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
       temp=DatasetLoading.loadDataNullable(path+"SelfRegulationSCP2Uni_TEST.arff");
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
      Instances temp=DatasetLoading.loadDataNullable(path+"FingerMovementsUni_TRAIN.arff");
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
      temp=DatasetLoading.loadDataNullable(path+"FingerMovementsUni_TEST.arff");
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
        Instances tr=DatasetLoading.loadDataNullable(arffP);
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
        Instances te=DatasetLoading.loadDataNullable(arffP2);
        System.out.println("Num instances ="+te.numInstances()+" num atts ="+te.numAttributes());
        multi=MultivariateInstanceTools.convertUnivariateToMultivariate(te,m);
      System.out.println("Num instances "+multi.numInstances());
      System.out.println("Num atts "+multi.numAttributes());
      arffTest=new OutFile("C:\\Users\\ajb\\Dropbox\\Data\\BCI Competition 3\\Data Set 1\\MotorImagery_TEST.arff");
      arffTest.writeLine(multi.toString());
        
        
        System.out.println("TEST Num instances ="+te.numInstances()+" num atts ="+te.numAttributes());
       
    }
    
    public static void formatERing(){
        InFile inf= new InFile("C:\\temp\\ERingTest.csv");
        InFile labs= new InFile("C:\\temp\\ERingTestLabels.csv");
        OutFile outf=new OutFile("C:\\temp\\ERing_TEST.arff");
        int[] labels=new int[270];
        int dims=4;
        for(int i=0;i<270;i++){
            labels[i]=labs.readInt();
            outf.writeString("'");
            for(int j=0;j<dims-1;j++){
                String temp=inf.readLine();
                System.out.println(temp);
                outf.writeString(temp+" \\n ");
            }
            String temp=inf.readLine();
            outf.writeLine(temp+"',"+labels[i]);
        }
         inf= new InFile("C:\\temp\\ERingTrain.csv");
         labs= new InFile("C:\\temp\\ERingTrainLabels.csv");
         outf=new OutFile("C:\\temp\\ERing_TRAIN.arff");
        labels=new int[30];
        for(int i=0;i<30;i++){
            labels[i]=labs.readInt();
            outf.writeString("'");
            for(int j=0;j<dims-1;j++){
                String temp=inf.readLine();
                System.out.println(temp);
                outf.writeLine(temp);
            }
            String temp=inf.readLine();
            outf.writeLine(temp+"',"+labels[i]);
        }    
    }
    
    
    
    public static void makeTSFormatFilesForUnivariate(String[] problems, String path, String outPath){
        boolean overwrite = true;
        for(int i=0;i<problems.length;i++){
//Check if they already exist            
            String prob=problems[i]; 
            System.out.println("PROBLEM "+prob);   
            Instances[] split= new Instances[2];
            split[0] =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
            split[1] =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");

            boolean univariate =true;
            for(int j=0;j<split[0].numAttributes()&& univariate; j++)
               if(split[0].attribute(j).isRelationValued())
                   univariate=false;
            if(!univariate){
                System.out.println("Problem "+prob+" is multivariate, call makeTSFormatFilesForMultivariate instead");
                continue;
            }
            OutFile[] tsFormat= new OutFile[2];
            
            File dir=new File(outPath+prob);
            if(!dir.isDirectory()){
                dir.mkdirs();
            }
            else if(!overwrite)
            {
                File f1=new File(outPath+prob+"\\"+prob+"_TEST.ts");
                File f2=new File(outPath+prob+"\\"+prob+"_TRAIN.ts");
                if(f1.exists() || f2.exists()){
                    System.out.println("Problem "+prob+ " already formatted, skipping");
                    continue;
                }
            }
            tsFormat[0]=new OutFile(outPath+prob+"\\"+prob+"_TRAIN.ts");
            tsFormat[1]=new OutFile(outPath+prob+"\\"+prob+"_TEST.ts");
            File f= new File(path+prob+"\\"+prob+".txt");
            if(!f.exists())
                System.out.println("ERROR cannot locate header "+path+prob+"\\"+prob+".txt");
            InFile comment= new InFile(path+prob+"\\"+prob+".txt");
    //Insert comment first 
            String line=comment.readLine();
            while(line!=null){
                if(tsFormat[0]!=null)
                    tsFormat[0].writeLine("#"+line);
                if(tsFormat[1]!=null)
                    tsFormat[1].writeLine("#"+line);
                line=comment.readLine();                
            }
            
            
            boolean padded=false; //Assume padded if last value is missing
            for(int j=0;j<split.length && !padded;j++){
                for(int k=0;k< split[j].numInstances()&& !padded; k++){
                   if(split[j].instance(k).isMissing(split[j].numAttributes()-2)){
                       padded=true;
                       System.out.println("Problem "+prob+" is padded for instance "+k+ " for split "+j);
                       System.out.println("Value = "+split[j].instance(k).value(split[j].numAttributes()-2)+" Previous value is "+split[j].instance(k).value(split[j].numAttributes()-3));
                    }
                }
            }
            boolean missing =false;
            if(!padded){ //Search the whole series
                for(int j=0;j<split.length && !missing;j++){
                    for(int k=0;k< split[j].numInstances() && !missing; k++){
                        if(split[j].instance(k).hasMissingValue())
                            missing=true;
                    }
                }
            }else{//, only search upto the last value present    
                for(int j=0;j<split.length && !missing;j++){
                    for(int k=0;k< split[j].numInstances() && !missing; k++){
//Find last value not missing
                        int end=split[j].numAttributes()-2;
                        while( end>=0 && split[j].instance(k).isMissing(end))
                            end--;
                        for(int m=0;m<=end;m++){
                           if(split[j].instance(k).isMissing(m)){
                               missing=true;
                               System.out.println("Case "+k+" missing value pos ="+m+" end = "+end); 
                               continue;
                           }
                        }
                    }
                }
            }
            System.out.println("Missing = "+missing); 
            for(int j=0;j<tsFormat.length;j++){
                tsFormat[j].writeLine("@problemName "+prob);
                tsFormat[j].writeLine("@timeStamps false");
                tsFormat[j].writeLine("@missing "+missing);
                tsFormat[j].writeLine("@univariate "+univariate);
                tsFormat[j].writeLine("@equalLength "+!padded);
                if(!padded)
                    tsFormat[j].writeLine("@seriesLength "+(split[j].instance(0).numAttributes()-1));
                    
                tsFormat[j].writeString("@classLabel true");
                Attribute cv= split[j].classAttribute();
      // Print the values of "position"
                Enumeration attValues = cv.enumerateValues();
                while (attValues.hasMoreElements()) {
                  String string = (String)attValues.nextElement();
                  tsFormat[j].writeString(" "+string);
                }
                tsFormat[j].writeLine("\n@data");
                System.out.println("Finished meta data");
                for(Instance ins:split[j]){
                    double[] data=ins.toDoubleArray();
//Find end if padded
                    int end=data.length-2;

                    if(padded){
                        while( end>=0 && Double.isNaN(data[end]))
                            end--;
                    }
//                        System.out.println("End of series = "+end);
                    if(Double.isNaN(data[0]))
                        tsFormat[j].writeString("?");
                    else
                        tsFormat[j].writeString(""+data[0]);
                    for(int k=1;k<=end;k++){
                        if(Double.isNaN(data[k]))
                            tsFormat[j].writeString(",?");
                        else
                            tsFormat[j].writeString(","+data[k]);
                    }
                    int classV=(int)data[data.length-1];
                    String clsString=cv.value(classV);
                    tsFormat[j].writeLine(":"+clsString);
                }
            }
        }
    }
    


    public static void makeTSFormatFilesForResamples(String[] problems, String path, String outPath){
        boolean overwrite = true;
        for(int i=0;i<problems.length;i++){
//Check if they already exist            
            String prob=problems[i]; 
            System.out.println("PROBLEM "+prob);   
            for(int fold=0;fold<30;fold++){
                Instances[] split= new Instances[2];
                split[0] =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+fold+"_TRAIN");
                split[1] =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+fold+"_TEST");
                boolean univariate =true;
                for(int j=0;j<split[0].numAttributes()&& univariate; j++)
                   if(split[0].attribute(j).isRelationValued())
                       univariate=false;
                if(!univariate){
                    System.out.println("Problem "+prob+" is multivariate, call makeTSFormatFilesForMultivariate intead");
                    continue;
                }
                OutFile[] tsFormat= new OutFile[2];
                File dir=new File(outPath+prob);
                if(!dir.isDirectory()){
                    dir.mkdirs();
                }
                else if(!overwrite)
                {
                    File f1=new File(outPath+prob+"\\"+prob+fold+"_TEST.ts");
                    File f2=new File(outPath+prob+"\\"+prob+fold+"_TRAIN.ts");
                    if(f1.exists() || f2.exists()){
                        System.out.println("Problem "+prob+ " already formatted, skipping");
                        continue;
                    }
                }
                tsFormat[0]=new OutFile(outPath+prob+"\\"+prob+fold+"_TRAIN.ts");
                tsFormat[1]=new OutFile(outPath+prob+"\\"+prob+fold+"_TEST.ts");
                File f= new File(path+prob+"\\"+prob+".txt");
                if(!f.exists())
                    System.out.println("ERROR cannot locate header "+path+prob+"\\"+prob+".txt");
                InFile comment= new InFile(path+prob+"\\"+prob+".txt");
        //Insert comment first 
                String line=comment.readLine();
                while(line!=null){
                    if(tsFormat[0]!=null)
                        tsFormat[0].writeLine("#"+line);
                    if(tsFormat[1]!=null)
                        tsFormat[1].writeLine("#"+line);
                    line=comment.readLine();                
                }
                boolean padded=false; //Assume padded if last value is missing
                for(int j=0;j<split.length && !padded;j++){
                    for(int k=0;k< split[j].numInstances()&& !padded; k++){
                       if(split[j].instance(k).isMissing(split[j].numAttributes()-2)){
                           padded=true;
                           System.out.println("Problem "+prob+" is padded for instance "+k+ " for split "+j);
                           System.out.println("Value = "+split[j].instance(k).value(split[j].numAttributes()-2)+" Previous value is "+split[j].instance(k).value(split[j].numAttributes()-3));
                        }
                    }
                }
                boolean missing =false;
                if(!padded){ //Search the whole series
                    for(int j=0;j<split.length && !missing;j++){
                        for(int k=0;k< split[j].numInstances() && !missing; k++){
                            if(split[j].instance(k).hasMissingValue())
                                missing=true;
                        }
                    }
                }else{//, only search upto the last value present    
                    for(int j=0;j<split.length && !missing;j++){
                        for(int k=0;k< split[j].numInstances() && !missing; k++){
    //Find last value not missing
                            int end=split[j].numAttributes()-2;
                            while( end>=0 && split[j].instance(k).isMissing(end))
                                end--;
                            for(int m=0;m<=end;m++){
                               if(split[j].instance(k).isMissing(m)){
                                   missing=true;
                                   System.out.println("Case "+k+" missing value pos ="+m+" end = "+end); 
                                   continue;
                               }
                            }
                        }
                    }
                }
                for(int j=0;j<tsFormat.length;j++){
                    tsFormat[j].writeLine("@problemName "+prob);
                    tsFormat[j].writeLine("@timeStamps false");
                    tsFormat[j].writeLine("@missing "+missing);
                    tsFormat[j].writeLine("@univariate "+univariate);
                    tsFormat[j].writeLine("@equalLength "+!padded);
                    if(!padded)
                        tsFormat[j].writeLine("@seriesLength "+(split[j].instance(0).numAttributes()-1));

                    tsFormat[j].writeString("@classLabel true");
                    Attribute cv= split[j].classAttribute();
          // Print the values of "position"
                    Enumeration attValues = cv.enumerateValues();
                    while (attValues.hasMoreElements()) {
                      String string = (String)attValues.nextElement();
                      tsFormat[j].writeString(" "+string);
                    }
                    tsFormat[j].writeLine("\n@data");
                    for(Instance ins:split[j]){
                        double[] data=ins.toDoubleArray();
    //Find end if padded
                        int end=data.length-2;

                        if(padded){
                            while( end>=0 && Double.isNaN(data[end]))
                                end--;
                        }
    //                        System.out.println("End of series = "+end);
                        if(Double.isNaN(data[0]))
                            tsFormat[j].writeString("?");
                        else
                            tsFormat[j].writeString(""+data[0]);
                        for(int k=1;k<=end;k++){
                            if(Double.isNaN(data[k]))
                                tsFormat[j].writeString(",?");
                            else
                                tsFormat[j].writeString(","+data[k]);
                        }
                        int classV=(int)data[data.length-1];
                        String clsString=cv.value(classV);
                        tsFormat[j].writeLine(":"+clsString);
                    }
                }
            }
        }
    }
    public static void makeTSFormatFilesForMultivariate(String[] problems, String path, String outPath, boolean overwrite){
        for(int i=0;i<problems.length;i++){
//Check if they already exist 
            System.out.println("Problem  = "+problems[i]);
            String prob=problems[i]; 
            File dir=new File(outPath+prob);
            if(!dir.isDirectory()){
                dir.mkdirs();
            }
            else if(!overwrite)
            {
                File f1=new File(outPath+prob+"\\"+prob+"_TEST.ts");
                File f2=new File(outPath+prob+"\\"+prob+"_TRAIN.ts");
                if(f1.exists() || f2.exists()){
                    System.out.println("Problem "+prob+ " already formatted, skipping");
                    continue;
                }
            }
            Instances[] split= new Instances[2];
            split[0] =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
            split[1] =DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
            int dimensions=split[0].instance(0).relationalValue(0).numInstances();
            boolean univariate =false;
            for(int j=0;j<split[0].numAttributes()-1&& !univariate; j++)
               if(!split[0].attribute(j).isRelationValued())
                   univariate=true;
            if(univariate){
                System.out.println("Problem "+prob+" is univariate, call makeTSFormatFilesForMultivariate intead. Skipping this one");
                continue;
            }
            System.out.println("PROBLEM "+prob+" has dimension "+dimensions);   
            OutFile[] tsFormat= new OutFile[2];
            
            tsFormat[0]=new OutFile(outPath+prob+"\\"+prob+"_TRAIN.ts");
            tsFormat[1]=new OutFile(outPath+prob+"\\"+prob+"_TEST.ts");
            File f= new File(path+prob+"\\"+prob+".txt");
            if(!f.exists())
                System.out.println("ERROR cannot locate header "+path+prob+"\\"+prob+".txt");
            InFile comment= new InFile(path+prob+"\\"+prob+".txt");
    //Insert comment first 
            String line=comment.readLine();
            while(line!=null){
                if(tsFormat[0]!=null)
                    tsFormat[0].writeLine("#"+line);
                if(tsFormat[1]!=null)
                    tsFormat[1].writeLine("#"+line);
                line=comment.readLine();                
            }
            
//Find if padded with missing values            
            boolean padded=false; //Assume padded if last value is missing
            for(int j=0;j<split.length && !padded;j++){
                for(int k=0;k< split[j].numInstances()&& !padded; k++){
                    Instances d=split[j].instance(k).relationalValue(0);
                    for(Instance ins:d){
                        if(ins.isMissing(d.numAttributes()-1)){
                            padded=true;
                            System.out.println("Problem "+prob+" is padded for instance "+k+ " for split "+j);
                            System.out.println("Split: "+j+" Case: "+k+" Value = "+ins.value(d.numAttributes()-1));
                            System.out.println("INSTANCE = "+ins);
                            break;
                        }
                    }
                }
            }
            int seriesLength=0;
            boolean missing =false;
            if(!padded){ //Search the whole series
                for(int j=0;j<split.length && !missing;j++){
                    for(int k=0;k< split[j].numInstances() && !missing; k++){
                        Instances d=split[j].instance(k).relationalValue(0); 
                        if(InstanceTools.hasMissing(d))
                            missing=true;
                    }
                }
                seriesLength=split[0].instance(0).relationalValue(0).numAttributes();
            }else{//, only search upto the last value present    
                for(int j=0;j<split.length && !missing;j++){
                    for(int k=0;k< split[j].numInstances() && !missing; k++){
                        Instances d=split[j].instance(k).relationalValue(0); 
                        for(int m=0;m<d.numInstances()&&!missing;m++){
                            int end=d.numAttributes()-1;
                            while( end>=0 && d.instance(m).isMissing(end))
                                end--;
                            for(int n=0;n<=end;n++){
                               if(d.instance(m).isMissing(n)){
                                   missing=true;
                                   System.out.println("Case "+k+" has missing value in dimension ="+m+" position = "+n+ "series length = "+end); 
                                   continue;
                               }
                            }
                        }
                    }
                }
            }
            System.out.println("Missing = "+missing); 
            for(int j=0;j<tsFormat.length;j++){
                tsFormat[j].writeLine("@problemName "+prob);
                tsFormat[j].writeLine("@timeStamps false");
                tsFormat[j].writeLine("@missing "+missing);
                tsFormat[j].writeLine("@univariate "+univariate);
                tsFormat[j].writeLine("@dimensions "+dimensions);
                tsFormat[j].writeLine("@equalLength "+!padded);
                if(!padded)
                tsFormat[j].writeLine("@seriesLength "+seriesLength);
                    
                tsFormat[j].writeString("@classLabel true");
                Attribute cv= split[j].classAttribute();
      // Print the values of "position"
                Enumeration attValues = cv.enumerateValues();
                while (attValues.hasMoreElements()) {
                  String string = (String)attValues.nextElement();
                  tsFormat[j].writeString(" "+string);
                }
                tsFormat[j].writeLine("\n@data");
                for(Instance ins:split[j]){
                    Instances dim=ins.relationalValue(0);
//Find end if padded
                    for(Instance d:dim){
                        int end=dim.numAttributes()-1;
                        double[] data=d.toDoubleArray();
                        if(padded){
                            while( end>=0 && Double.isNaN(data[end]))
                            end--;
                        }
                        if(Double.isNaN(data[0]))
                            tsFormat[j].writeString("?");
                        else
                            tsFormat[j].writeString(""+data[0]);
                        for(int k=1;k<=end;k++){
                            if(Double.isNaN(data[k]))
                                tsFormat[j].writeString(",?");
                            else
                                tsFormat[j].writeString(","+data[k]);
                        }
                        tsFormat[j].writeString(":");
                    }
                    int classV=(int)ins.classValue();
                    String clsString=cv.value(classV);
                    tsFormat[j].writeLine(clsString);
                }
            }
        }
    }
   //<editor-fold defaultstate="collapsed" desc="Multivariate TSC datasets 2018 release">    
    public static String[] mtscProblems2018={
        "InsectWingbeat",//15
//        "KickVsPunch", Poorly formatted and very small train size
        "JapaneseVowels",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",//20
        "PenDigits",
        "PEMS-SF",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",//25
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "StandWalkJump",        
        "UWaveGestureLibrary"            
};    
       //</editor-fold>       
    public static void main(String[] args) throws Exception {
        generateAllResamplesInARFF();
        System.exit(0);
        
        String[] paths={"E:\\ArchiveData\\Multivariate_arff\\","E:\\ArchiveData\\Multivariate_ts\\"};
        String dest="E:\\ArchiveData\\Zips_Multivariate\\";
        String[] probs={"AsphaltObstaclesCoordinates","AsphaltPavementTypeCoordinates","AsphaltRegularityCoordinates"};
//        makeZips(probs,dest,paths);
        
        paths=new String[]{"E:\\ArchiveData\\Univariate_arff\\","E:\\ArchiveData\\Univariate_ts\\"};
        dest="E:\\ArchiveData\\Zips_Univariate\\";
        probs=DatasetLists.tscProblems2018;
//        makeZips(probs,dest,paths);
        makeTSFormatFilesForResamples(probs,paths[0],paths[1]);
        System.exit(0);
//        makeTSFormatFilesForUnivariate(probs,path,outPath);
//        path="E:/ArchiveData/Multivariate_arff/";
//        outPath="E:/ArchiveData/Multivariate_ts/";
//        makeTSFormatFilesForMultivariate(probs,path,outPath,true);

        System.exit(0);
        
 //       checkAllZipFiles();
//       summariseUnivariateData("E:\\ShapeletPaper\\Version 2\\Transforms\\ShapeletTransform1\\Transforms\\");
       summariseUnivariateData("E:\\ShapeletPaper\\Version 2\\Transforms\\ShapeletTransform10\\Transforms\\");
//       summariseUnivariateData("Z:\\Data\\TSCProblems2018\\");
       makeAllZipFiles();
//formatERing();
//        String path="C:\\temp\\";
//        Instances test = DatasetLoading.loadDataNullable(path+"ERing_TEST");
//        Instances train = DatasetLoading.loadDataNullable(path+"ERing_TRAIN");
//        Instances train2 = DatasetLoading.loadDataNullable(path+"ERing_TRAIN2");
//insertNewLineSpaces();
//checkSpeechMarks();
//removeSpeechMarks();
//System.out.println("Summarise data ");
testSimpleClassifier();
//        summariseData();
//        formatDuckDuckGeese();        
//        formatCricket();
         summariseMultivariateData();
        debugFormat();
        makeConcatenatedFiles();
 /*       String prob="UWaveGestureLibrary";
        String dest="Z:\\Data\\UnivariateMTSC\\";
        String path="Z:\\Data\\Multivariate TSC Problems\\";
        Instances test = DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
        Instances train = DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
        
     Instances test2 = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TEST");
  Instances train2 = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TRAIN");
   */      
//        summariseData();
//        System.exit(0);
//        debugFormat();
//        makeUnivariateFiles();
//        String prob="UWaveGestureLibrary";
//        String dest="Z:\\Data\\UnivariateMTSC\\";
//        String path="Z:\\Data\\Multivariate TSC Problems\\";
//        Instances test = DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TEST");
//    Instances train = DatasetLoading.loadDataNullable(path+prob+"\\"+prob+"_TRAIN");
//        
//       Instances test2 = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TEST");
//  Instances train2 = DatasetLoading.loadDataNullable(dest+prob+"\\"+prob+"_TRAIN");
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
        train=DatasetLoading.loadDataNullable("Z:\\Data\\MultivariateTSCProblems\\ECGActivities\\ECGActivities_TRAIN");
        test=DatasetLoading.loadDataNullable("Z:\\Data\\MultivariateTSCProblems\\ECGActivities\\ECGActivities_TEST");
        
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
        x=DatasetLoading.loadDataNullable(sourcePath+"EpilepsyX_ALL");
        y=DatasetLoading.loadDataNullable(sourcePath+"EpilepsyY_ALL");
        z=DatasetLoading.loadDataNullable(sourcePath+"EpilepsyZ_ALL");
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
        Instances train = DatasetLoading.loadDataNullable(path);
        System.out.println(" univariate data = "+train);
        path="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Multivariate\\multivariateConcatExample";
        train = DatasetLoading.loadDataNullable(path);
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


    public static void generateAllResamplesInARFF(){
        //       String path = "C:\\Users\\ajb\\Dropbox\\Working docs\\Data Resample Debug\\Data\\";
        String path = "Z:\\ArchiveData\\";
        String[] datasets=DatasetLists.tscProblems2018;
//        datasets=new String[]{"Chinatown"};
        for(String problem:datasets){
            System.out.println("Generating folds for "+problem);
            Instances[] data = { null, null };
            File trainFile = new File(path + problem + "/" + problem + "_TRAIN.arff");
            File testFile = new File(path + problem + "/" + problem + "_TEST.arff");
            boolean predefinedFold0Exists = (trainFile.exists() && testFile.exists());
            if (predefinedFold0Exists) {
                Instances train = DatasetLoading.loadDataNullable(trainFile);
                Instances test = DatasetLoading.loadDataNullable(testFile);
                for(int fold =0;fold<=29;fold++){
                    data[0] = new Instances(train);  //making absolutely sure no funny business happening
                    data[1] = new Instances(test);

                    if (train.checkForAttributeType(Attribute.RELATIONAL))
                        data = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances(data[0], data[1], fold);
                    else
                        data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], fold);
                    System.out.println(" instance 0 in fold "+fold+" train "+data[0].instance(0).toString());

                    //toString produces 'printing-friendly' 6 sig figures for doubles, using proper arffsaver now instead
                    DatasetLoading.saveDataset(data[0], path + problem + "/" + problem + fold+"_TRAIN.arff");
                    DatasetLoading.saveDataset(data[1], path + problem + "/" + problem + fold+"_TEST.arff");

//                    //Save folds.
//                    OutFile trainF=new OutFile(path + problem + "/" + problem + fold+"_TRAIN.arff");
//                    trainF.writeLine(data[0].toString());
//                    OutFile testF=new OutFile(path + problem + "/" + problem + fold+"_TEST.arff");
//                    testF.writeLine(data[1].toString());

                }
            }else{
                System.out.println("File does not exist on "+path);
            }

        }

    }
    
    
}
