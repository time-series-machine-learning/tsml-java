/*
class to collate standard results files over multiple classifiers and problems

Usage 
(assuming Collate.jar has this as the main class): 
java -jar Collate.jar ResultsDirectory/ ProblemDirectory/ NumberOfFolds Classifier1 Classifier2 .... ClassifierN NoParasC1 NoParasC2 .... NoParasCn
e.g. java -jar -Xmx6000m Collate.jar Results/ UCIContinuous/ 30 RandF RotF 2 2 

collates the results for 30 folds for RandF and RotF in the directory for Results 
on all the problems in UCIContinous (as defined by having a directory in the folder)

Stage 1: take all the single fold files, work out the diagnostics on test data: 
Accuracy, BalancedAccuracy, NegLogLikelihood, AUROC and F1 and store the TrainCV accuracy. 
all done by call to collateFolds();
Combine folds into a single file for each statistic in ResultsDirectory/ClassifierName
these are
Counts: counts.csv, number per problem (max number is NumberOfFolds, it does not check for more).
Diagnostics: TestAcc.csv, TestF1.csv, TestBAcc.csv, TestNLL.csv, TestAUROC.csv, TrainCVAcc.csv
Timings: Timings.csv
Parameter info: Parameter1.csv, Parameter2.csv...AllTuningAccuracies.csv (if tuning occurs, all tuning values).

Stage 2: 
Output: Classifier Summary: call to method averageOverFolds() 
Creates average and standard deviation over all folds based on the files created at stage 1 with the addition of the mean difference
per fold.
All put in a single directory.

Stage 3
Final Comparison Summary: call to method basicSummaryComparisons();        
a single file in ResultsDirectory directory called summaryTests<ClassifierNames>.csv
contains pairwise comparisons of all the classifiers. 

1. All Pairwise Comparisons for TestAccDiff, TestAcc, TestBAcc, TestNLL.csv and TestAUROC

1. Wins/Draws/Loses
2. Mean (and std dev) difference
3. Two sample tests of the mean values 







 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import statistics.distributions.BinomialDistribution;
import statistics.tests.OneSampleTests;
import utilities.ClassifierResults;

/**
 *
 * @author ajb
 */
public class CollateResults {
    public static File[] dir;
    static String basePath;
    static String[] classifiers;
    static ArrayList<String> problems;
    static String problemPath;
    static int folds;
    static int numClassifiers;
    static int[] numParas;
    static DecimalFormat df=new DecimalFormat("##.######");
    static double[][] data;
    static boolean countPartials=false;
  
/** 
 * Arguments: 
1. ResultsDirectory/ 
2. Either ProblemDirectory/ or ProblemFiles.csv or ProblemFiles.txt 
*                       Basically checks for an extension and if its there reads a file. 
* 
3. NumberOfFolds 
4-4+nosClassifiers                          Classifier1 Classifier2 .... ClassifierN 
4+nosClassifiers+1 to 4+2*nosClassifiers    NoParasC1 NoParasC2 .... NoParasCn
 * 
 */
    public static void readData(String[] args){
        int numInitArgs=4;
        basePath=args[0];
        System.out.println("Base path = "+basePath);
        problemPath=args[1];
        System.out.println("Problem path = "+problemPath);
         folds =Integer.parseInt(args[2]);
        System.out.println("Folds = "+folds);
        String partial=args[3].toLowerCase();        
        if(partial.equals("true"))
            countPartials=true;
        else
            countPartials=false;
                
        numClassifiers=(args.length-numInitArgs)/2;
        classifiers=new String[numClassifiers];
        for(int i=0;i<classifiers.length;i++)
            classifiers[i]=args[i+numInitArgs];
        numParas=new int[classifiers.length];
        for(int i=0;i<classifiers.length;i++)
            numParas[i]=Integer.parseInt(args[i+numInitArgs+classifiers.length]);
//Get problem files
        File f=new File(problemPath);
            problems=new ArrayList<>();
        if(problemPath.contains(".txt") || problemPath.contains(".csv")){//Read from file
            if(!f.exists())
                System.out.println("Error loading problems  from file ="+problemPath);
            else{
                InFile inf=new InFile(problemPath);
                String prob=inf.readLine();
                while(prob!=null){
                    problems.add(prob);
                    prob=inf.readLine();
                }
            }
        }
        else{
            if(!f.isDirectory()){
                System.out.println("Error in problem path ="+problemPath);
            }

            dir=f.listFiles();
            for(File p:dir){
                if(p.isDirectory()){
                    problems.add(p.getName());
                }
            }
        }
        Collections.sort(problems);

    }
/*Returns True if the file is present and correct
    Changed cos it is too slow at the moment
    */    
    public static boolean validateSingleFoldFile(String str){
        File f= new File(str);
        if(f.exists()){ // Check 1: non zero
             if(f.length()==0){//Empty, delete file
                 f.delete();
             }
             else{
                 try{
/*                    InFile inf=new InFile(str);
                    int c=inf.countLines();
                    if(c<=3){//No  predictions, delete
                        inf.closeFile();
                        f.delete();
                        return false;
                    }
                    inf.closeFile();
  */                  return true; 
                 }catch(Exception e){
                     System.out.println("Exception thrown trying to read file "+str);
                     System.out.println("Exception = "+e+"  THIS MAY BE A GOTCHA LATER");
                     e.printStackTrace();
                     return false;
                 }
//Something in there, it is up to ClassifierResults to validate the rest
             }
        }
        return false;
    }
/**
 * Stage 1: take all the single fold files, work out the diagnostics on test data: 
Accuracy, BalancedAccuracy, NegLogLikelihood, AUROC and F1 and store the TrainCV accuracy. 
all done by call to collateFolds();
Combine folds into a single file for each statistic in ResultsDirectory/ClassifierName
these are
Counts: counts.csv, number per problem (max number is NumberOfFolds, it does not check for more).
Diagnostics: TestAcc.csv, TestF1.csv, TestBAcc.csv, TestNLL.csv, TestAUROC.csv, TrainCVAcc.csv
Timings: Timings.csv
Memory: Memory.csv
Parameter info: Parameter1.csv, Parameter2.csv...AllTuningAccuracies.csv (if tuning occurs, all tuning values).

 */    
    public static int MAXNUMPARAS=1180;
    public static void collateFolds(){
//        String[] allStats={"TestAcc","TrainCVAcc","TestNLL","TestBACC","TestAUROC","TestF1"};      

        for(int i=0;i<classifiers.length;i++){
            String cls=classifiers[i];
            System.out.println("Processing classifier ="+cls);
            File f=new File(basePath+cls);
            if(f.isDirectory()){ //Check classifier directory exists.
                System.out.println("Base path "+basePath+cls+" exists");
                File stats=new File(basePath+cls+"/SummaryStats");
                if(!stats.isDirectory())
                    stats.mkdir();
                String filePath=basePath+cls+"/SummaryStats/";
                OutFile clsResults=new OutFile(filePath+cls+"TestAcc.csv");
                OutFile f1Results=new OutFile(filePath+cls+"TestF1.csv");
                OutFile BAccResults=new OutFile(filePath+cls+"TestBAcc.csv");
                OutFile nllResults=new OutFile(filePath+cls+"TestNLL.csv");
                OutFile AUROCResults=new OutFile(filePath+cls+"TestAUROC.csv");
                OutFile trainResults=new OutFile(filePath+cls+"TrainCVAcc.csv");
                OutFile[] paraFiles=new OutFile[numParas[i]];
                for(int j=0;j<paraFiles.length;j++)
                    paraFiles[j]=new OutFile(filePath+cls+"Parameter"+(j+1)+".csv");
                OutFile timings=new OutFile(filePath+cls+"Timings.csv");
                OutFile mem=new OutFile(filePath+cls+"Memory.csv");
                OutFile allAccSearchValues=new OutFile(filePath+cls+"AllTuningAccuracies.csv");
                OutFile missing=null;
                OutFile counts=new OutFile(filePath+cls+"Counts.csv");
                OutFile partials=null;
                if(countPartials)
                        partials=new OutFile(filePath+cls+"PartialCounts.csv");;
                OutFile of = new OutFile(filePath+cls+"Corrupted.csv");
                int missingCount=0;
                for(String name:problems){            
//Write collated results for this classifier to a single file                
                    OutFile mergedResults=new OutFile(filePath+cls+"AllTestPrediction"+name+".csv");
                    clsResults.writeString(name);
                    trainResults.writeString(name);
                    f1Results.writeString(name);
                    BAccResults.writeString(name);
                    nllResults.writeString(name);
                    AUROCResults.writeString(name);
                    allAccSearchValues.writeString(name);
                    timings.writeString(name);
                    mem.writeString(name);
                    for(OutFile out:paraFiles)
                        out.writeString(name+",");
//GAVIN HACK                    
//                    String path=basePath+cls+"/"+name+"/results/";
                    String path=basePath+cls+"//Predictions//"+name;
                    if(missing!=null && missingCount>0)
                        missing.writeString("\n");
                    missingCount=0;
                    if(countPartials)
                        partials.writeString(name);
                    int caseCount=0;
                    for(int j=0;j<folds;j++){
    //Check fold exists and is a valid file
                        boolean valid=validateSingleFoldFile(path+"//testFold"+j+".csv");

                        if(valid){
//This could fail if file only has partial probabilities on the line
    //Read in test accuracy and store                    
    //Check fold exists
    //Read in test accuracy and store
                            InFile inf=null;
                            String[] trainRes=null;
                            try{
                                inf=new InFile(path+"//testFold"+j+".csv");
                                inf.readLine();
                                trainRes=inf.readLine().split(",");//Stores train CV and parameter info
                                clsResults.writeString(","+inf.readDouble());
                                if(trainRes.length>1){//There IS parameter info
                                    //First is train time build
                                    String str=trainRes[1].trim();
                                    timings.writeString(","+df.format(Double.parseDouble(str)));
                                    //second is the trainCV testAcc
                                    if(trainRes.length>3){
                                        str=trainRes[3].trim();
                                        trainResults.writeString(","+Double.parseDouble(str));
                                        //Then variable list of numParas
                                        int pos=5;
                                        for(int k=0;k<numParas[i];k++){
                                            if(trainRes.length>pos){
                                                paraFiles[k].writeString(trainRes[pos]+",");
                                                pos+=2;    
                                            }
                                            else
                                                paraFiles[k].writeString(",");
                                        }
    //                                    write the rest to the para search file
                                        while(pos<trainRes.length)
                                            allAccSearchValues.writeString(","+trainRes[pos++]);    
                                    }
                                }
                                else{
                                    trainResults.writeString(",");
                                    for(int k=0;k<numParas[i];k++)
                                        paraFiles[k].writeString(",");
                                }
//Read in the rest into a ClassifierResults object
                                inf.closeFile();
//                                inf.openFile(path+"//testFold"+j+".csv");
  //                              int temp=(inf.countLines()-3);
    //                            inf.closeFile();
  //                              System.out.println("Number of items in bag "+(j+1)+" = "+temp);
//                                caseCount+=temp;
                                ClassifierResults res=new ClassifierResults();
                                res.loadFromFile(path+"//testFold"+j+".csv");
                                mergedResults.writeLine(res.writeInstancePredictions());                                
                                res.findAllStats();
                                f1Results.writeString(","+res.f1);
                                BAccResults.writeString(","+res.balancedAcc);
                                nllResults.writeString(","+res.nll);
                                AUROCResults.writeString(","+res.meanAUROC);
                                
                            }catch(Exception e){
                                System.out.println(" Error "+e+" in "+path);
                                if(trainRes!=null){
                                    System.out.println(" second line read has "+trainRes.length+" entries :");
                                    for(String str:trainRes)
                                        System.out.print(str+",");
                                    System.out.println("XX"+trainRes[1]+"XX AND TRIMMED: XX"+trainRes[1].trim()+"XX");
                                    of.writeLine(name+","+j);
                                    e.printStackTrace();
                                    System.exit(1);
                                }
                            }finally{
                                if(inf!=null)
                                    inf.closeFile();

                            }
                            if(countPartials)
                               partials.writeString(",0");
                        }
                        else{
                            if(missing==null)
                                missing=new OutFile(filePath+cls+"MissingFolds.csv");
                            if(missingCount==0)
                                missing.writeString(name);
                            missingCount++;
                           missing.writeString(","+j);
                           if(countPartials){
    //Fold j missing, count here how many parameters are complete on it                           
                               int x=0;
                               for(int k=1;k<MAXNUMPARAS;k++){
                                   if(validateSingleFoldFile(path+"//fold"+j+"_"+k+".csv"))
                                       x++;
                               }
                                if(countPartials)
                                    partials.writeString(","+x);
                           }
                        }
                    }
//                    System.out.println(" Total number of cases ="+caseCount);
                    counts.writeLine(name+","+(folds-missingCount));
                    if(countPartials)
                        partials.writeString("\n");
                    clsResults.writeString("\n");
                    trainResults.writeString("\n");
                    f1Results.writeString("\n");
                    BAccResults.writeString("\n");
                    nllResults.writeString("\n");
                    AUROCResults.writeString("\n");
                    timings.writeString("\n");
                    allAccSearchValues.writeString("\n");
                    
                    for(int k=0;k<paraFiles.length;k++)
                        paraFiles[k].writeString("\n");
                }
                clsResults.closeFile();
                trainResults.closeFile();
                    for(int k=0;k<paraFiles.length;k++)
                        paraFiles[k].closeFile();
            }
            else{
                System.out.println("Classifier "+cls+" has no results directory: "+basePath+cls);
                System.out.println("Exit ");
                System.exit(0);
            }
        }
        
    }

/** Stage 2: 
Output: Classifier Summary: call to method averageOverFolds() 
Creates average and standard deviation over all folds based on the files created at stage 1 with the addition of the mean difference
per fold.
All put in a single directory.
* **/
public static void averageOverFolds(){
        
        String name=classifiers[0];
        for(int i=1;i<classifiers.length;i++)
            name+=classifiers[i];
       String filePath=basePath+name+"/";
        if(classifiers.length==1)
            filePath+="SummaryStats/";
        File nf=new File(filePath);
        if(!nf.isDirectory())
            nf.mkdirs();
        String[] allStats={"MeanTestAcc","MeanTrainCVAcc","MeanTestNLL","MeanTestBAcc","MeanTestAUROC","MeanTestF1","MeanTimings"};      
        String[] testStats={"TestAcc","TrainCVAcc","TestNLL","TestBAcc","TestAUROC","TestF1","Timings"};      
        OutFile[] means=new OutFile[allStats.length];
        for(int i=0;i<means.length;i++)
            means[i]=new OutFile(filePath+allStats[i]+name+".csv"); 
        OutFile[] stDev=new OutFile[allStats.length];
        for(int i=0;i<stDev.length;i++)
            stDev[i]=new OutFile(filePath+allStats[i]+"StDev"+name+".csv"); 
        OutFile count=new OutFile(filePath+"Counts"+name+".csv");

//Headers        
        for(int i=0;i<classifiers.length;i++){
            for(OutFile of:means)
                of.writeString(","+classifiers[i]);
            for(OutFile of:stDev)
                of.writeString(","+classifiers[i]);
            count.writeString(","+classifiers[i]);
        }
        for(OutFile of:means)
            of.writeString("\n");
        for(OutFile of:stDev)
            of.writeString("\n");
        count.writeString("\n");
//Do counts first
        InFile[] allClassifiers=new InFile[classifiers.length];
        for(int i=0;i<allClassifiers.length;i++){
            String str=basePath+classifiers[i]+"/SummaryStats/"+classifiers[i];
            System.out.println("Loading "+str+"Counts.csv");
            String p=str+"Counts.csv";
            if(new File(p).exists())
                allClassifiers[i]=new InFile(p);
            else{
                allClassifiers[i]=null;//superfluous
                System.out.println("File "+p+" does not exist");
            }
        }
        for(String str:problems){
            count.writeString(str);
            for(int i=0;i<allClassifiers.length;i++){
                if(allClassifiers[i]!=null){
                   allClassifiers[i].readString();
                   count.writeString(","+allClassifiers[i].readInt());
                }
                else{
                    count.writeString(",");
                }
            }
            count.writeString("\n");

        }
        
        for(int j=0;j<allStats.length;j++){
//Open files with data for all folds        
            for(int i=0;i<allClassifiers.length;i++){
                String str=basePath+classifiers[i]+"/SummaryStats/"+classifiers[i];
                String p=str+testStats[j]+".csv";
                if(new File(p).exists())
                    allClassifiers[i]=new InFile(p);
                else{
                    allClassifiers[i]=null;//superfluous
                                    System.out.println("File "+p+" does not exist");
                }
            }
//Find means             
            for(String str:problems){
                means[j].writeString(str);
                stDev[j].writeString(str);
                String prev="First";
                for(int i=0;i<allClassifiers.length;i++){
                    if(allClassifiers[i]==null){
                        means[j].writeString(",");
                        stDev[j].writeString(",");
                    }
                    else{//Find mean
                        try{
                            String r=allClassifiers[i].readLine();
                            String[] res=r.split(",");
                            double mean=0;
                            double sumSquare=0;
                            for(int m=1;m<res.length;m++){
                                double d=Double.parseDouble(res[m].trim());
                                mean+=d;
                                sumSquare+=d*d;
                            }
                            if(res.length>1){
                                int size=(res.length-1);
                                mean=mean/size;
                                double stdDev=sumSquare/size-mean*mean;
                                stdDev=Math.sqrt(stdDev);
                                means[j].writeString(","+df.format(mean));
                                stDev[j].writeString(","+df.format(stdDev));
                            }
                            else{
                                means[j].writeString(",");
                                stDev[j].writeString(",");
                            }
                            prev=r;
                        }catch(Exception ex){
                            System.out.println("failed to read line: "+ex+" previous line = "+prev+" file index ="+j+" classifier index ="+i);
                        }
                    }        
                }
                means[j].writeString("\n");
                stDev[j].writeString("\n");
                if(j==0)
                    count.writeString("\n");
             }
            for(InFile  inf:allClassifiers)
                if(inf!=null)
                    inf.closeFile();
        }
    }

public static void basicSummaryComparisons(){
//Only compares first two
    DecimalFormat df = new DecimalFormat("###.#####");
    if(classifiers.length<=1)
        return;
    String name=classifiers[0];
    for(int i=1;i<classifiers.length;i++)
        name+=classifiers[i];
    OutFile s=new OutFile(basePath+"summaryTests"+name+".csv");
    String[] allStatistics={"TestAcc","TestBAcc","TestNLL","TestAUROC"};
    data=new double[problems.size()][classifiers.length];
    s.writeLine(name);
    for(String str:allStatistics){
        s.writeLine("**************"+str+"********************");
        System.out.println("Loading "+basePath+name+"/"+str+name+".csv");
        InFile f=new InFile(basePath+name+"/"+str+name+".csv");
        f.readLine();
        for(int i=0;i<problems.size();i++){
            String ss=f.readLine();
            String[] d=ss.split(",");
            for(int j=0;j<classifiers.length;j++)
                data[i][j]=-1; 

            for(int j=0;j<d.length-1;j++){
                    try{
                    double v=Double.parseDouble(d[j+1]);
                    data[i][j]=v;
                    }catch(Exception e){
// yes yes I know its horrible, but this is text parsing, not rocket science
//                            System.out.println("No entry for classifier "+j);
                    }
            }
//                for(int j=0;j<classifiers.length;j++)
//                    System.out.println(" Classifier "+j+" has data "+data[i][j]);       
        }
        for(int x=0;x<classifiers.length-1;x++){
            for (int y=x+1; y < classifiers.length; y++) {//Compare x and y
                int wins=0,draws=0,losses=0;
                int sigWins=0,sigLosses=0;
                double meanDiff=0;
                double sumSq=0;
                double count=0;
                for(int i=0;i<problems.size();i++){
                    if(data[i][x]!=-1 && data[i][y]!=-1){
                        if(data[i][x]>data[i][y])
                            wins++;
                        else if(data[i][x]==data[i][y])
                            draws++;
                        else
                            losses++;
                        meanDiff+=data[i][x]-data[i][y];
                        sumSq+=(data[i][x]-data[i][y])*(data[i][x]-data[i][y]);
                        count++;
                    }
                }
//                    DecimalFormat df = new DecimalFormat("##.#####");
                System.out.println(str+","+classifiers[x]+","+classifiers[y]+",WIN/DRAW/LOSE,"+wins+","+draws+","+losses);  
                BinomialDistribution bin=new BinomialDistribution();
                bin.setParameters(wins+losses,0.5);
                double p=bin.getCDF(wins);
                if(p>0.5) p=1-p;
                s.writeLine(str+","+classifiers[x]+","+classifiers[y]+",WIN/DRAW/LOSE,"+wins+","+draws+","+losses+", p =,"+df.format(p));  
                System.out.println(str+","+classifiers[x]+","+classifiers[y]+",COUNT,"+count+",MeanDiff,"+df.format(meanDiff/count)+",StDevDiff,"+df.format((sumSq-(meanDiff*meanDiff)/count))+" p ="+df.format(p));  
        //3. Find out how many are statistically different within folds
//Do paired T-tests from fold files
            InFile first=new InFile(basePath+classifiers[x]+"/"+classifiers[x]+str+".csv");
            InFile second=new InFile(basePath+classifiers[y]+"/"+classifiers[y]+str+".csv");
            for(int i=0;i<problems.size();i++){
//Read in both: Must be the same number to proceed
                String[] probX=first.readLine().split(",");
                String[] probY=second.readLine().split(",");
                if(probX.length<=folds || probY.length<=folds)
                    continue;   //Skip this problem
                double[] diffs=new double[folds];
                boolean notAllTheSame=false;
                for(int j=0;j<folds;j++){
                    diffs[j]=Double.parseDouble(probX[j+1])-Double.parseDouble(probY[j+1]);
                    if(!notAllTheSame && !probX[j+1].equals(probY[j+1]))
                        notAllTheSame=true;
                }

                if(notAllTheSame){
                    OneSampleTests test=new OneSampleTests();
                    String res=test.performTests(diffs);
                    System.out.println("Results = "+res);
                    String[] results=res.split(",");
                    double tTestPValue=Double.parseDouble(results[2]);
                    if(tTestPValue>=0.95) sigWins++;
                    else if(tTestPValue<=0.05) sigLosses++;
                }
                else
                    System.out.println("**************ALL THE SAME problem = "+probX[0]+" *************");
          }
            s.writeLine(str+","+classifiers[x]+","+classifiers[y]+",SIGWIN/SIGLOSS,"+sigWins+","+sigLosses);  
            System.out.println(str+","+classifiers[x]+","+classifiers[y]+",SIGWIN/SIGLOSS,"+sigWins+","+sigLosses);  
        //2. Overall mean difference
            s.writeLine(str+","+classifiers[x]+","+classifiers[y]+",COUNT,"+count+",MeanDiff,"+df.format(meanDiff/count)+",StDevDiff,"+df.format((sumSq-(meanDiff*meanDiff)/count)));  
            System.out.println(str+","+classifiers[x]+","+classifiers[y]+",COUNT,"+count+",MeanDiff,"+df.format(meanDiff/count)+",StDevDiff,"+df.format((sumSq-(meanDiff*meanDiff)/count)));  
        }


    }

//Do pairwise tests over all common datasets. 
//1.    First need to condense to remove any with one missing
        ArrayList<double[]> res=new ArrayList<>();
        for(int i=0;i<data.length;i++){
            int j=0;
            while(j<data[i].length && data[i][j]!=-1)
                j++;
            if(j==data[i].length)
                res.add(data[i]);
        }
        System.out.println("REDUCED DATA SIZE = "+res.size());

        double[][] d2=new double[res.size()][];
        for(int i=0;i<res.size();i++)
            d2[i]=res.get(i);

 //2. Do pairwise tests       
        StringBuilder resultsString=MultipleClassifiersPairwiseTest.runSignRankTest(d2,classifiers);
        s.writeString(resultsString.toString());
        System.out.println(resultsString);

    }
    s.closeFile();
}
    
    public static void collate(String[] args){
//STAGE 1: Read from arguments, find problems        
       readData(args);
        System.out.println(" number of classifiers ="+numClassifiers);
 //STAGE 2: Collate the individual fold files into one        
        System.out.println("Collating folds ....");
        collateFolds();
        System.out.println("Collate folds finished. \n Averaging over folds....");
       
//STAGE 3: Summarise over folds 
        averageOverFolds();
        System.out.println("averaging folds finished.\n Basic stats comparison ....");
//STAGE 4: Do statical comparisons
        basicSummaryComparisons();        
        
    }
   public static void untunedVsTuned() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//UCI//Analysis//", "XBBoost", 5);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.UCIContinuousWithoutBigFour, 0, 117)); 
       m.readInClassifiers(new String[] {"XGBoost"},
               "E://Results//UCI//Untuned");
       m.readInClassifiers(new String[] {"TunedXGBoost"}, 
               "E://Results//UCI//Tuned");
           m.runComparison(); 
       
   }

   public static void collateTuned() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//UCI//Analysis//", "Tuned", 5);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.UCIContinuousWithoutBigFour, 0, 117)); 
       m.readInClassifiers(new String[] {"MLP2","SVMRBF","SVMP","RandF","RotF","XGBoost"}, 
               "E://Results/UCI/Tuned");
       
       
//       m.readInClassifiers(new String[] {"TunedSVMPolynomial","TunedSVMRBF","TunedXGBoost","TunedMLP","TunedSingleLayerMLP","TunedTWoLayerMLP","TunedRandF","TunedRotF","RotF"}, 
//               "E://Results//UCI//Tuned");
       m.runComparison(); 

       
   }
      public static void collateRotFSensitivity() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//UCI//Analysis//", "RotFGroupSize", 30);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.UCIContinuousWithoutBigFour, 0, 117)); 
       m.readInClassifiers(new String[] {
           "RotFG3","RotFG4","RotFG5","RotFG6","RotFG7","RotFG8","RotFG9","RotFG10","RotFG11","RotFG12"}, 
               "E://Results/UCI/RotFSize");
       
       
//       m.readInClassifiers(new String[] {"TunedSVMPolynomial","TunedSVMRBF","TunedXGBoost","TunedMLP","TunedSingleLayerMLP","TunedTWoLayerMLP","TunedRandF","TunedRotF","RotF"}, 
//               "E://Results//UCI//Tuned");
       m.runComparison(); 

       
   }

      public static void collateRotFSensitivity2() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//UCI//Analysis//", "RotFProp", 30);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.UCIContinuousWithoutBigFour, 0, 117)); 
       m.readInClassifiers(new String[] {
           "RotRP1","RotRP2","RotRP3","RotRP4","RotRP5","RotRP6","RotRP7","RotRP8","RotRP9"}, 
               "E://Results/UCI/RotFPercentRemoved");
       
       
//       m.readInClassifiers(new String[] {"TunedSVMPolynomial","TunedSVMRBF","TunedXGBoost","TunedMLP","TunedSingleLayerMLP","TunedTWoLayerMLP","TunedRandF","TunedRotF","RotF"}, 
//               "E://Results//UCI//Tuned");
       m.runComparison(); 

       
   }

   public static void collateUntuned() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(
               "E://Results//UCI//Analysis//", "RandF10000", 30);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.UCIContinuousFileNames, 0, 121)); 
       
       m.readInClassifiers(new String[] {"RandF","RotF","RandF10000"},//"SVMRBF","UBMLP 
               "E://Results//UCI//Untuned");

       m.runComparison(); 

       
   }
   public static void ucrComparison() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//UCR//Analysis//", "RotFvsRandF", 30);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.tscProblems85, 0, 85)); 
       m.readInClassifiers(new String[] {"RotF","RandF"},"E://Results//UCR//Untuned");
//       m.readInClassifiers(new String[] {"DTWCV"},"E://Results//UCR//Tuned");
       m.setTestResultsOnly(true);
           m.runComparison(); 
       
   }
   public static void stucrComparison() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//STUCR//Analysis//", "STRotFvsRandF", 30);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.tscProblems85, 0, 85)); 
       m.readInClassifiers(new String[] {"RotF","RandF","SVMQ"},"E://Results//STUCR");
//       m.readInClassifiers(new String[] {"DTWCV"},"E://Results//UCR//Tuned");
       m.setTestResultsOnly(true);
           m.runComparison(); 
       
   }

   public static void tunedVuntuned() throws Exception{
       String classifier ="SVMRBF";
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//UCI//Analysis//", classifier, 30);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.UCIContinuousFileNames, 0, 121)); 
       m.readInClassifiers(new String[] {classifier},"E://Results//UCI//Untuned");
       m.readInClassifiers(new String[] {"Tuned"+classifier},"E://Results//UCI//Tuned");
               
               
               
               //"//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/FinalisedUCIContinuous/");
       m.setTestResultsOnly(true);
           m.runComparison(); 
       
   }
    public static void collateBags() {
         int folds=45;
         String[] classifiers={"BOSS","DTWCV","ED","EE","RandF","RISE","RotF","ST","TSF"};
         for(String str:classifiers){   
            String source="E:/Results/Bags/BagsTwoClassHistogramProblem/"+str+"/Predictions/BagsTwoClassHistogramProblem";
            String dest="E:/Results/Bags/BagsTwoClassHistogramProblem/"+str+"/Predictions/BagsTwoClassHistogramProblem";
            OutFile outfTest=new OutFile(dest+"/"+str+"TestAll.csv");
            OutFile outfTrain=new OutFile(dest+"/"+str+"TrainAll.csv");
            for(int i=0;i<folds;i++){
                System.out.println("Formatting "+str+" fold "+i);
                InFile infTest=new InFile(source+"/testFold"+i+".csv");
                InFile infTrain=new InFile(source+"/trainFold"+i+".csv");
                String line = infTest.readLine();
                line = infTest.readLine();
                line = infTest.readLine();
                line = infTest.readLine();
                while(line!=null){
                    outfTest.writeLine(line);
                    line = infTest.readLine();
                }
                 line = infTrain.readLine();
                 line = infTrain.readLine();
                 line = infTrain.readLine();
                 line = infTrain.readLine();
                while(line!=null){
                    outfTrain.writeLine(line);
                    line = infTrain.readLine();
                }
            }
        }
    }  
   public static void bagsStats() throws Exception{
       MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//STUCR//Analysis//", "PCA", 45);
       m.setBuildMatlabDiagrams(true);
       m.setDebugPrinting(true);
       m.setUseAllStatistics();
       m.setDatasets(Arrays.copyOfRange(development.DataSets.tscProblems85, 0, 85)); 
       m.readInClassifiers(new String[] {"RotF","RandF","SVMQ"},"E://Results//STUCR");
//       m.readInClassifiers(new String[] {"DTWCV"},"E://Results//UCR//Tuned");
       m.setTestResultsOnly(true);
           m.runComparison(); 
       
   } 
//First argument: String path to results directories
//Second argument: path to directory with problem allStats to look for
//Third argument: number of folds    
//Next x arguments: x Classifiers to collate    
//Next x arguments: number of numParas stored for each classifier    
    public static void main(String[] args) throws Exception {
 //collateRotFSensitivity();
 //collateRotFSensitivity2();
 //System.exit(0);
//collateBags();
 //ucrRotFvsRandFtestOnly();
//        collateUntuned();
//       collateTuned();
//     ucrComparison();
//        jamesStats();
//  reformatUBMLP();
//ucrComparison();
//stucrComparison();
//BMLPtunedVuntuned();
//untunedVsTuned();

// System.exit(0);

//NOTE TO SELF
//Below is using my stats generator not james. To use James put in a static
//method and exit, as above
//    String[] classifiers={"BOSS","CAWPE","CAWPE_AS_COTE","ED","RandF","RotF","SLOWDTWCV","ST","TSF","RISE","XGBoost"};
    String[] classifiers={"TunedSVMQuad"};
    for(String classifier:classifiers){
        String parameters="1";
        if(args.length>1)
            collate(args);
        else{ 
            String[] str={"Z:\\BagsSDM\\Results\\",
                "Z:\\BagsSDM\\Data\\","45","false",classifier,parameters};
            collate(str);
        }
    }
}
    public static void reformatUBMLP()//Insert an extra comma
    {
        int folds=30;
        String source="E:\\Results\\UCI\\Untuned\\UBMLP_OLD\\Predictions";
        String dest="E:\\Results\\UCI\\Untuned\\UBMLP\\Predictions";
        
        for(String str:DataSets.UCIContinuousFileNames){
            for(int i=0;i<folds;i++){
                System.out.println("Formatting "+str+" fold "+i);
                InFile infTest=new InFile(source+"/"+str+"/testFold"+i+".csv");
                InFile infTrain=new InFile(source+"/"+str+"/trainFold"+i+".csv");
                File out=new File(dest+"/"+str);
                if(!out.isDirectory())
                    out.mkdirs();
                OutFile outfTest=new OutFile(dest+"/"+str+"/testFold"+i+".csv");
                OutFile outfTrain=new OutFile(dest+"/"+str+"/trainFold"+i+".csv");
                for(int j=0;j<3;j++){
                    outfTest.writeLine(infTest.readLine());
                    outfTrain.writeLine(infTrain.readLine());
                }
                String line = infTest.readLine();
                while(line!=null){
                    String[] split=line.split(",");
                    outfTest.writeString(split[0]+","+split[1]+",");
                    for(int j=2;j<split.length;j++)
                        outfTest.writeString(","+split[j]);
                    outfTest.writeString("\n");
                    line = infTest.readLine();
                }
                 line = infTrain.readLine();
                while(line!=null){
                    String[] split=line.split(",");
                    outfTrain.writeString(split[0]+","+split[1]+",");
                    for(int j=2;j<split.length;j++)
                        outfTrain.writeString(","+split[j]);
                    outfTrain.writeString("\n");
                    line = infTrain.readLine();
                }
            }
                    
        }
        
    }    
}
