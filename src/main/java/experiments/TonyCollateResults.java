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
package experiments;

import evaluation.MultipleClassifiersPairwiseTest;
import evaluation.MultipleClassifierEvaluation;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import statistics.distributions.BinomialDistribution;
import statistics.tests.OneSampleTests;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLists;
import java.util.HashMap;

/**
 * Class to collate results from any classifier creating standard output 
 * There are two ways to collate results. 
 * 1. (Tony Bagnall) The code in this class creates summary info for individual classifiers. 
 * It does not do comparisons between classifiers, and it will build with incomplete
 * data, ignoring incomplete data sets. This can be run on the cluster (see below). 
 * See method individualClassifiersCollate() for example usage
 * 2 (James Large) Using the MultipleClassifierEvaluation class, detailed  
 * comparisons between classifier can be conducted. This can create matlab driven
 * critical difference diagrams

**On the cluster usage:**
* Class to collate standard results files over multiple classifiers and problems
* Usage 
* (assuming Collate.jar has this as the main class): 
* java -jar Collate.jar ResultsDirectory/ ProblemDirectory/ NumberOfFolds Classifier1 Classifier2 .... ClassifierN NoParasC1 NoParasC2 .... NoParasCn
* e.g. java -jar -Xmx6000m Collate.jar Results/ UCIContinuous/ 30 RandF RotF 2 2 

* collates the results for 30 folds for RandF and RotF in the directory for Results 
* on all the problems in UCIContinous (as defined by having a directory in the folder)
* How it works:
* 
* Stage 1: take all the single fold files, work out the diagnostics on test data: 
* Accuracy, BalancedAccuracy, NegLogLikelihood, AUROC and F1 and store the TrainCV accuracy. 
* all done by call to collateFolds();
* Combine folds into a single file for each statistic in ResultsDirectory/ClassifierName
* these are
* Counts: counts.csv, number per problem (max number is NumberOfFolds, it does not check for more).
* Diagnostics: TestAcc.csv, TestF1.csv, TestBAcc.csv, TestNLL.csv, TestAUROC.csv, TrainCVAcc.csv, Timings.csv
* Parameter info: Parameter1.csv, Parameter2.csv...AllTuningAccuracies.csv (if tuning occurs, all tuning values).
* 
* Stage 2: 
* Output: Classifier Summary: call to method averageOverFolds() 
* Creates average and standard deviation over all folds based on the   
* created at stage 1 with the addition of the mean difference per fold. All put in a single directory.
* 
* Stage 3
* Final Comparison Summary: call to method basicSummaryComparisons();        
* a single file in ResultsDirectory directory called summaryTests<ClassifierNames>.csv
* contains pairwise comparisons of all the classifiers. 
* 1. All Pairwise Comparisons for TestAccDiff, TestAcc, TestBAcc, TestNLL.csv and TestAUROC
*   1. Wins/Draws/Loses
*   2. Mean (and std dev) difference
*   3. Two sample tests of the mean values 
*
* @author ajb
**/
public class TonyCollateResults {
    public static File[] dir;
    static String basePath;
    static String[] classifiers;
    public static ArrayList<String> problems;
    static boolean readProblemNamesFromDir=true;
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
//Get problem files from a directory if required
        if(readProblemNamesFromDir){
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
                OutFile[] paraFiles=null;
                if(numParas[i]>0){
                    paraFiles=new OutFile[numParas[i]];
                    for(int j=0;j<paraFiles.length;j++)
                        paraFiles[j]=new OutFile(filePath+cls+"Parameter"+(j+1)+".csv");
                }
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
                    if(numParas[i]>0){
                        for(OutFile out:paraFiles)
                            out.writeString(name+",");
                    }
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
                                inf.readLine();//First line, problem info
                                trainRes=inf.readLine().split(",");//Stores train CV and parameter info
                                
                                clsResults.writeString(","+inf.readDouble());
                                timings.writeString(","+inf.readDouble());   

                                if(trainRes.length>1){//There IS parameter info
                                    int pos=1;
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
                                else{
//                                    trainResults.writeString(",");
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
                                res.loadResultsFromFile(path+"//testFold"+j+".csv");
                                mergedResults.writeLine(res.instancePredictionsToString());                                
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
                    
                    for(int k=0;k<numParas[i];k++)
                        paraFiles[k].writeString("\n");
                }
                clsResults.closeFile();
                trainResults.closeFile();
                    for(int k=0;k<numParas[i];k++)
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
        StringBuilder resultsString= MultipleClassifiersPairwiseTest.runSignRankTest(d2,classifiers);
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
/**
 * 
 *First argument: String path to results directories
 * Second argument: path to directory with problem allStats to look for
 * Third argument: number of folds    
 * Next x arguments: x Classifiers to collate    
 * Next x arguments: number of numParas stored for each classifier    
   **/
   public static void singleClassifiersFullStats(String[] args) throws Exception{
        if(args.length>1)
            collate(args);
        
        else{ 
        String[] classifiers={"TSF"};//,"EE","RISE","ST","BOSS"};
        for(String classifier:classifiers){
            String parameters="0";
            String[] str={"E:\\Results\\Bakeoff Redux\\",
                "Z:\\ArchiveData\\Univariate_arff\\","30","false",classifier,parameters};
//Change this to read an array            
            collate(str);
        }
    }

   }
   public static void shapeletResultsSummary() throws Exception{
        String[] problems={""};
        String[] classifiers={""};
        String readPath="";
//Where to put them, directory name, number of folds           
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//somewhere", "OneHour", 1);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        m.setDatasets(problems);
        m.readInClassifiers(classifiers, readPath); 
        m.runComparison(); 
       
       
   }

    
/**
 * Usage of MultipleClassifierEvaluation. See the class for more info
 * @throws Exception 
 */
   public static void multipleClassifierFullStats(String[] args) throws Exception{
       if(args.length>0){
//TO DO           


       }
       else{ //Example manual setting
           
           
//Where to put them, directory name, number of folds           
            MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Results//UCI//Analysis//", "Tuned", 5);
            m.setBuildMatlabDiagrams(true);
            m.setDebugPrinting(true);
            m.setUseAllStatistics();
            m.setDatasets(Arrays.copyOfRange(experiments.data.DatasetLists.UCIContinuousWithoutBigFour, 0, 117));
            m.readInClassifiers(new String[] {"MLP2","SVMRBF","SVMP","RandF","RotF","XGBoost"}, 
                    "E://Results/UCI/Tuned");
            m.runComparison(); 
       }
   }
   public static final String bakeOffPathBeast="Z:/ReferenceResults/CollatedResults/Bakeoff2015/byClassifier/";
   public static final  String hiveCotePathBeast="Z:/ReferenceResults/CollatedResults/HIVE-COTE2017/";
   public static final  String reduxPathBeast="Z:/ReferenceResults/CollatedResults/BakeoffRedux2019/";
   public static final  String bakeOffPathCluster="/gpfs/home/ajb/Results/ReferenceResults/Bakeoff2015/ByClassifier/";
   public static final  String hiveCotePathCluster="/gpfs/home/ajb/Results/ReferenceResults/HIVE-COTE2017/";
   public static final  String reduxPathCluster="/gpfs/home/ajb/Results/ReferenceResults/BakeoffRedux2019/";

   public static String bakeOffPath=bakeOffPathBeast;
   public static String hiveCotePath=hiveCotePathBeast;
   public static String reduxPath=reduxPathBeast;
   



   
   
/**
 * Quick in place collation and comparison to reference results 
 * @param args
 * @throws Exception 
 * Primary: these results are built from file using predictions or just reading line 3 
 * para 1: String: location of primary results, including classifier name
 * para 2: Boolean stored as string: true: calculate acc from preds and check. False: just read from line 3. 
 * Para 3: Integer stored as string: number of folds to look for
 * OPTIONAL: these results are read directly, can have as many as desired 
 * Input format ProblemSource,ClassifierName. Problem source must be Bakeoff, HIVE-COTE, or Redux 
 * para 3: comparison classifier TYPE,NAME 1
 * para 4: comparison classifier full path 2
 * ..
 * Notes: 
 * 1. Only uses accuracy, does not require classes map 0... numClasses-1 or probabilities.
 * 2. Assumes file structure is arg[0]/Predictions/ProblemName/testFold0.csv
 * 3. Assumes every directory in Predictions is a results folder
 * 4. For the fold averages, it ignores any problem without a full set of results, will print the results as empty
 * 5. Prints results to arg[0]/QuickResults/TrainTest<classifierName>.csv, 
 */
    public static void singleClassifiervsReferenceResults(String[] args) throws Exception{
        if(args.length<4){
            String input="";
            for(String s:args)
                input+=s+" ";
            throw new Exception("Wrong input args =:"+input);
        }
        for(int i=0;i<args.length;i++){
            System.out.println("args["+i+"] = "+args[i]);
        }
        String fullPath=args[0];
        String[] temp=args[0].split("/");
        String classifierName=temp[temp.length-1];
        System.out.println(" Primary Classifier = "+classifierName);
        boolean calcAcc=Boolean.parseBoolean(args[1]);
        System.out.println(" Recalculate accuracies from file? = "+calcAcc);
        folds=Integer.parseInt(args[2]);
        System.out.println("Folds = "+folds);
        boolean oldFormat=Boolean.parseBoolean(args[3]);
        System.out.println(" Use old format ? = "+oldFormat);
        File f= new File(fullPath+"/QuickResults");
        f.mkdirs();
//Get primary results
        ArrayList<String> problems =new ArrayList<>();
        ArrayList<String> missing =new ArrayList<>();
        f=new File(fullPath+"/Predictions");
        System.out.println(fullPath+"/Predictions");
        File[] fileList=f.listFiles();
        System.out.println("File names in "+fullPath+"/Predictions  : has "+f.length()+" files ");
        for(File t:fileList){
            System.out.println("\t"+t.getName());

            if(t.isDirectory()){ // Note 3: assume all dirs are problems
                problems.add(t.getName());
            }
        }
        Collections.sort(problems);
        
        double[] trainTest= new double[problems.size()];
        double[] trainTestTime= new double[problems.size()];
        double[] means= new double[problems.size()];
        double[][] allFolds= new double[problems.size()][folds];
        double[] meansTime= new double[problems.size()];
        double[][] allFoldsTime= new double[problems.size()][folds];
        OutFile trTsFile=new OutFile(fullPath+"/QuickResults/TrainTest"+classifierName+".csv");
        OutFile meansFile=new OutFile(fullPath+"/QuickResults/Average"+folds+classifierName+".csv");
        OutFile allFoldsFile=new OutFile(fullPath+"/QuickResults/AllFolds"+classifierName+".csv");
        OutFile trTsTimesFile=new OutFile(fullPath+"/QuickResults/TimesTrainTest"+classifierName+".csv");
        OutFile meanTimesFile=new OutFile(fullPath+"/QuickResults/TimeAverage"+folds+classifierName+".csv");
        OutFile allFoldsTimesFile=new OutFile(fullPath+"/QuickResults/TimeAllFolds"+classifierName+".csv");
        OutFile trainFileCount=new OutFile(fullPath+"/QuickResults/trainFileCount"+classifierName+".csv");
              
        InFile inf=null;
        boolean readTimes=true;
        for(int i=0;i<trainTest.length;i++){
            System.out.println("Processing "+problems.get(i));
            int trainCount=0;
            boolean cont=true;
            for(int j=0;j<folds && cont;j++){
                try{
                    inf=new InFile(fullPath+"/Predictions/"+problems.get(i)+"/testFold"+j+".csv");
                    inf.readLine();//Ignore first two
                    String secondLine=inf.readLine();
                    String[] split;
                    String[] secondSplit=secondLine.split(",");
                    String thirdLine=inf.readLine();
                    String[] thirdSplit=thirdLine.split(",");
//Under the old format, the time is the second argument of line 2
//Under the new format, the time is 
                    double time=0;
                    try{
                        if(oldFormat){
                            time=Double.parseDouble(secondSplit[1]);
                        }else{
                            time=Double.parseDouble(thirdSplit[1]);
                        }
                        
                    }catch(Exception e){
                        System.out.println("Error reading in times for base classifier, oldFormat="+oldFormat+" may be wrong");
                        System.out.println("Continue without timing");
                        readTimes=false;
                    }
                    
                    double acc=Double.parseDouble(thirdSplit[0]);//
                    
                    if(calcAcc){
                        String line=inf.readLine();
                        double a=0;
                        int count=0;
                        while(line!=null){
                            split=line.split(",");
                            count++;
                            if(split[0].equals(split[1]))
                                a++;
                            line=inf.readLine();
                        }
                        if(count>0)
                            a/=count;
                        if((a-acc)>0.000000001){
                            System.out.println("Mismatch in acc read from file and acc calculated from file");
                            System.out.println("THIS NEEDS INVESTIGATING. Abandoning the whole problem compilation ");
                            System.exit(1);
                        }
                    }
                    if(j==0){
                        trainTest[i]=acc;
                        trainTestTime[i]=time;
                    }
                    allFolds[i][j]=acc;
                    allFoldsTime[i][j]=time;
                    File tr_f=new File(fullPath+"/Predictions/"+problems.get(i)+"/trainFold"+j+".csv");
                    if(tr_f.exists()){//Train fold present
                        trainCount++;
                    }
                    
                    
                }catch(Exception e){
                    missing.add(problems.get(i));
                    System.out.println("Some error processing "+fullPath+"/Predictions/"+problems.get(i)+"/testFold"+j+".csv");
                    System.out.println(" Abandoning entire problem "+problems.get(i));
                    cont=false;
                }
                finally{
                    if(inf!=null)
                        inf.closeFile();
                }
            }
            if(cont){//Should have all the data
                trTsFile.writeString(problems.get(i));
                meansFile.writeString(problems.get(i));
                allFoldsFile.writeString(problems.get(i));
                trainFileCount.writeLine(problems.get(i)+","+trainCount);
                trTsFile.writeString(","+trainTest[i]);
                means[i]=0;
                for(int j=0;j<allFolds[i].length;j++){
                    allFoldsFile.writeString(","+allFolds[i][j]);
                    means[i]+=allFolds[i][j];
                }                
                means[i]/=folds;
                meansFile.writeString(","+means[i]);
                trTsFile.writeString("\n");
                meansFile.writeString("\n");
                allFoldsFile.writeString("\n");
                if(readTimes){
                    trTsTimesFile.writeString(problems.get(i));
                    meanTimesFile.writeString(problems.get(i));
                    allFoldsTimesFile.writeString(problems.get(i));
                    trTsTimesFile.writeString(","+trainTestTime[i]);
                    meansTime[i]=0;
                    for(int j=0;j<allFolds[i].length;j++){
                        allFoldsTimesFile.writeString(","+allFoldsTime[i][j]);
                        meansTime[i]+=allFoldsTime[i][j];
                    }                
                    meansTime[i]/=folds;
                    meanTimesFile.writeString(","+meansTime[i]);
                    trTsTimesFile.writeString("\n");
                    meanTimesFile.writeString("\n");
                    allFoldsTimesFile.writeString("\n");
                }

            }else{//Write trainTest if present
                if(trainTest[i]>0){ //Captured fold 0, lets use it
                    trTsFile.writeLine(problems.get(i)+","+trainTest[i]);
                    if(readTimes)
                        trTsTimesFile.writeString(problems.get(i));
                }
            }
        }        
        if(args.length>4){ //Going to compare to some others
            String[] rc=new String[args.length-4];
            for(int i=4;i<args.length;i++)
                rc[i-4]=args[i];
            System.out.println("Comparing "+classifierName+" to ");
            String[][] classifiers=new String[rc.length][];
            for(int i=0;i<rc.length;i++)
                classifiers[i]=rc[i].split(",");
            ArrayList<HashMap<String,Double>> trainTestResults=new ArrayList<>();
            ArrayList<HashMap<String,Double>> averageResults=new ArrayList<>();
            for(int i=0;i<classifiers.length;i++){
                classifiers[i][0]=classifiers[i][0].toUpperCase();
                System.out.println(classifiers[i][0]+"_"+classifiers[i][1]);
                HashMap<String,Double> trTest=new HashMap<>();
                HashMap<String,Double> averages=new HashMap<>();
//Look for train results
                String path="";
                switch(classifiers[i][0]){
                    case "BAKEOFF":
                        path=bakeOffPath;
                        break;
                    case "HIVE-COTE":
                        path=hiveCotePath;
                        break;
                    case "REDUX":
                        path=reduxPath;
                        break;
                    default: 
                        System.out.println("UNKNOWN LOCATION INDICATOR "+classifiers[i][0]);
                        throw new Exception("UNKNOWN LOCATION INDICATOR "+classifiers[i][0]);
                }
                f=new File(path+"TrainTest/TrainTest"+classifiers[i][1]+".csv");
                if(f.exists()){
                    inf=new InFile(path+"TrainTest/TrainTest"+classifiers[i][1]+".csv");
                    String line=inf.readLine();
                    while(line!=null){
                        String[] split=line.split(",");
                        String prob=split[0];
                        if(prob.equals("CinCECGtorso"))//Hackhackityhack: legacy problem
                            prob="CinCECGTorso";
                        if(prob.equals("StarlightCurves"))//Hackhackityhack: legacy problem
                            prob="StarLightCurves";
                        if(prob.equals("NonInvasiveFatalECGThorax1"))//Hackhackityhack: legacy problem
                            prob="NonInvasiveFetalECGThorax1";
                        if(prob.equals("NonInvasiveFatalECGThorax2"))//Hackhackityhack: legacy problem
                            prob="NonInvasiveFetalECGThorax2";
                        
                        Double d = Double.parseDouble(split[1]);
                        trTest.put(prob,d);
                        line=inf.readLine();
                    }
                }
                f=new File(path+"Average30/Average30"+classifiers[i][1]+".csv");
                if(f.exists()){
                    inf=new InFile(path+"Average30/Average30"+classifiers[i][1]+".csv");
//                    inf.readLine();
                    String line=inf.readLine();
                    while(line!=null){
                        String[] split=line.split(",");
                        String prob=split[0];
                        if(prob.equals("CinCECGtorso"))//Hackhackityhack: legacy problem
                            prob="CinCECGTorso";
                        if(prob.equals("StarlightCurves"))//Hackhackityhack: legacy problem
                            prob="StarLightCurves";
                        if(prob.equals("NonInvasiveFatalECGThorax1"))//Hackhackityhack: legacy problem
                            prob="NonInvasiveFetalECGThorax1";
                        if(prob.equals("NonInvasiveFatalECGThorax2"))//Hackhackityhack: legacy problem
                            prob="NonInvasiveFetalECGThorax2";
                        Double d = Double.parseDouble(split[1]);
                        averages.put(prob,d);
                        line=inf.readLine();
                    }
                }
                trainTestResults.add(trTest);
                averageResults.add(averages);
            }
            trTsFile=new OutFile(fullPath+"/QuickResults/CompareTrainTest"+classifierName+".csv");
            OutFile trTsFileComplete=new OutFile(fullPath+"/QuickResults/CompareTrainTestCompleteOnly"+classifierName+".csv");
            meansFile=new OutFile(fullPath+"/QuickResults/CompareAverage"+folds+"_"+classifierName+".csv");
            OutFile meansFileComplete=new OutFile(fullPath+"/QuickResults/CompareAverageCompleteOnly"+classifierName+".csv");
            trTsFile.writeString("Problem,"+classifierName);
            meansFile.writeString("Problem,"+classifierName);
            meansFileComplete.writeString("Problem,"+classifierName);
            trTsFileComplete.writeString("Problem,"+classifierName);
            for(int i=0;i<classifiers.length;i++){
                trTsFile.writeString(","+classifiers[i][0]+"_"+classifiers[i][1]);
                meansFile.writeString(","+classifiers[i][0]+"_"+classifiers[i][1]);
                meansFileComplete.writeString(","+classifiers[i][0]+"_"+classifiers[i][1]);
                trTsFileComplete.writeString(","+classifiers[i][0]+"_"+classifiers[i][1]);
            }
            trTsFile.writeString("\n");
            meansFile.writeString("\n");
            meansFileComplete.writeString("\n");
            trTsFileComplete.writeString("\n");
            for(int i=0;i<problems.size();i++){
                String name=problems.get(i);
                boolean present=true;
//Train test                
                if(trainTest[i]>0){ //Captured fold 0, lets use it
                    String line=name+","+trainTest[i];
                    trTsFile.writeString(name+","+trainTest[i]);
                    for(int j=0;j<classifiers.length;j++){
                        HashMap<String,Double> trTest=trainTestResults.get(j);
                        if(trTest.containsKey(name)){
                            Double x=trTest.get(name);
                            trTsFile.writeString(","+x);
                            line+=","+x;
                        }
                        else{
                             trTsFile.writeString(",");
                             present=false;
                        }
                    }
                    trTsFile.writeString("\n");
                    if(present)
                        trTsFileComplete.writeLine(line);
                }
//Averages
                if(!missing.contains(name)){
                    String line=name+","+means[i];
                    meansFile.writeString(name+","+means[i]);
                    for(int j=0;j<classifiers.length;j++){
                        HashMap<String,Double> av=averageResults.get(j);
                        if(av.containsKey(name)){
                            Double x=av.get(name);
                            meansFile.writeString(","+x);
                            line+=","+x;
                        }
                        else{
                             meansFile.writeString(",");
                             present=false;
                        }
                    }
                    meansFile.writeString("\n");
                    if(present)
                        meansFileComplete.writeLine(line);
                }
            }
            
        }        
        
        
    }
    static String[] makeQuickStatsArgs(String primary, boolean calcAcc, int folds, String...others) throws Exception{
        String[] input;
        if(others==null)
            input=new String[4];
        else 
            input=new String[4+others.length];
        input[0]=primary;
        input[1]=calcAcc+"";
        input[2]=folds+"";
        input[3]="false";
        if(others!=null)
            for(int i=0;i<others.length;i++)
                input[i+4]=others[i];
        return input;
    }
    public static void quickStats(String primary, boolean calcAcc, int folds, String...others) throws Exception{
        String[] input;
        if(others==null)
            input=new String[3];
        else 
            input=new String[3+others.length];
        input[0]=primary;
        input[1]=calcAcc+"";
        input[2]=folds+"";
        if(others!=null)
            for(int i=0;i<others.length;i++)
                input[i+3]=others[i];
       singleClassifiervsReferenceResults(input);
        
    }
    public static void oneOffCollate() throws Exception{
        
            String[] classifierList={"RotF_Josh","RotF_Markus"};//sktime1","sktime2"};//,"RotF","DTWCV"};{"PythonTSF","PythonTSFComposite",
            String readPath="E://Temp/Python/";

//Where to put them, directory name, number of folds
            MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E:/Temp/RotFDebug/","rotf_debug",1);
            m.setIgnoreMissingResults(true);
            m.setBuildMatlabDiagrams(true);
            m.setDebugPrinting(true);
            m.setUseAllStatistics();
            m.setDatasets(DatasetLists.ReducedUCI);
            m.setTestResultsOnly(true);
            m.readInClassifiers(classifierList,readPath);
            classifierList=new String[]{"RotF"};
            readPath="E://Temp/";
            m.readInClassifiers(classifierList,readPath);
            m.runComparison(); 


    }
    
    public static void rotFDebug() throws Exception{
            String[] classifierList={"RotFMarkus","RotF"};//sktime1","sktime2"};//,"RotF","DTWCV"};{"PythonTSF","PythonTSFComposite",
            String readPath="Z://RotFDebug/";
            MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("Z:/RotFDebug/","markus_no_norm",30);
            m.setIgnoreMissingResults(true);
            m.setBuildMatlabDiagrams(true);
            m.setDebugPrinting(true);
            m.setUseAllStatistics();
            String[] allProbs=new String[DatasetLists.ReducedUCI.length+DatasetLists.tscProblems112.length];
            System.arraycopy(DatasetLists.ReducedUCI,0,allProbs,0,DatasetLists.ReducedUCI.length);
            System.arraycopy(DatasetLists.tscProblems112,0,allProbs,DatasetLists.ReducedUCI.length,DatasetLists.tscProblems112.length);
            m.setDatasets(allProbs);
            m.setTestResultsOnly(true);
            m.readInClassifiers(classifierList,readPath);
   //         classifierList=new String[]{"RandF","RotF"};//sktime1","sktime2"};//,"RotF","DTWCV"};{"PythonTSF","PythonTSFComposite",
 //           readPath="E://Results/RotFDebug/UCINorm/";
//            m.readInClassifiers(classifierList,readPath);
            m.runComparison(); 
   }

    public static void bakeOffRedux() throws Exception{
        String[] classifierList={"EE","BOSS","TSF","RISE","STC"};
                //"DTWCV","STC","EE","BOSS","TSF","RISE","HIVE-COTE"};//sktime1","sktime2"};//,"RotF","DTWCV"};{"PythonTSF","PythonTSFComposite",
        String readPath="Z:/ReferenceResults/Hive Cote Components REDUX/";

//Where to put them, directory name, number of folds
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,"bakeoff_redux",30);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems78;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        m.readInClassifiers(classifierList,readPath);
         classifierList=new String[]{"FlatCote"};//sktime1","sktime2"};//,"RotF","DTWCV"};{"PythonTSF","PythonTSFComposite",
           readPath="E:/Results Working Area/Hybrids/";
           m.readInClassifiers(classifierList,readPath);
        m.runComparison();



    }

    public static void compareDistanceBased(int folds, boolean testOnly) throws Exception {
        String readPath="E:/Results Working Area/DistanceBased/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "distance_compare_"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        m.setTestResultsOnly(testOnly);

        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] benchmark={"EE","PF","FastEE"};
        m.readInClassifiers(benchmark,readPath);
        m.runComparison();
    }

    public static void compareDictionary(int folds, boolean testOnly) throws Exception {

        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E:/Results Working Area/DictionaryBased/",
                "dictionary_compare_with_csBOSS",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(false);
        m.setUseAllStatistics();
//        String[] allProbs=DatasetLists.newProblems27;
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(testOnly);
        String readPath="E:/Results Working Area/DictionaryBased/";
        String[] classifierList={"BOSS","S-BOSS","WEASEL","cBOSS","cS-BOSS"};//,};//,"
        m.readInClassifiers(classifierList,readPath);
        m.runComparison();
        boolean recalAcc=false;
        //These for quick stats
        bakeOffPath=bakeOffPathBeast;
        hiveCotePath=hiveCotePathBeast;
        reduxPath= reduxPathBeast;
        for(String cls:classifierList) {
            //                    String comparisons=null;
            String[] comparisons = {"HIVE-COTE,BOSS", "HIVE-COTE,HIVE-COTE"};
            String[] args = makeQuickStatsArgs(readPath + cls, recalAcc, folds, comparisons);
            singleClassifiervsReferenceResults(args);
        }

        problems=new ArrayList<>();
        readProblemNamesFromDir=false;
        for(String str:allProbs)
            problems.add(str);
        for(String cls:classifierList) {
            String parameters = "0";
            String[] str = {readPath,
                    "Z:\\ArchiveData\\Univariate_arff\\", folds + "", "false", cls, parameters};
            //Change this to read an array
            collate(str);
        }
    }

    public static void compareHiveCoteVariants() throws Exception {
        int folds=30;
        String readPath="E:/Results Working Area/HC Variants/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "hc_compare_"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems85;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] newClassifiers={"HC-Alpha4","HC-Alpha1","HC-Catch22TSF"};
        m.readInClassifiers(newClassifiers,readPath);
   //     readPath="E:/Results Working Area/Hybrids/";
     //   String[] c2={"TSCHIEF","HiveCote"};
     //   m.readInClassifiers(c2,readPath);


        m.runComparison();
    }
    public static void compareShapeletVariants(int folds) throws Exception {

        String readPath="E:/Results Working Area/STC Variants/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "shapelet_compare_"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        //"k200","k300","k400","k600","k700","k800","k900",
        String[] classifierList={"k100","k500","k1000","hr1000k100"};//,"BOSS","SpatialBOSS","cBOSS"};//,"cBOSS",
        m.readInClassifiers(classifierList,readPath);

        m.readInClassifiers(new String[]{"STC"},"Z:\\ReferenceResults\\Hive Cote Components REDUX\\");

        m.runComparison();
        boolean recalAcc=false;
        //These for quick stats
        bakeOffPath=bakeOffPathBeast;
        hiveCotePath=hiveCotePathBeast;
        reduxPath= reduxPathBeast;
        for(String cls:classifierList) {
            //                    String comparisons=null;
            String[] comparisons = {"HIVE-COTE,ST", "HIVE-COTE,HIVE-COTE"};
            String[] args = makeQuickStatsArgs(readPath + cls, recalAcc, folds, comparisons);
            singleClassifiervsReferenceResults(args);
        }

        problems=new ArrayList<>();
        readProblemNamesFromDir=false;
        for(String str:allProbs)
            problems.add(str);
        for(String cls:classifierList) {
            String parameters = "0";
            String[] str = {readPath,
                    "Z:\\ArchiveData\\Univariate_arff\\", folds + "", "false", cls, parameters};
            //Change this to read an array
            collate(str);
        }
    }



    public static void compareShapelets(int folds) throws Exception {

        String readPath="E:/Results Working Area/STC Variants/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "shapelet_compare_"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] classifierList={"STC-Bakeoff","STC-DAWAK","ShapeletTreeClassifier"};//,"BOSS","SpatialBOSS","cBOSS"};//,"cBOSS",
        m.readInClassifiers(classifierList,readPath);
        m.runComparison();
        boolean recalAcc=false;
        //These for quick stats
        bakeOffPath=bakeOffPathBeast;
        hiveCotePath=hiveCotePathBeast;
        reduxPath= reduxPathBeast;
        for(String cls:classifierList) {
            //                    String comparisons=null;
            String[] comparisons = {"HIVE-COTE,ST", "HIVE-COTE,HIVE-COTE"};
            String[] args = makeQuickStatsArgs(readPath + cls, recalAcc, folds, comparisons);
            singleClassifiervsReferenceResults(args);
        }

        problems=new ArrayList<>();
        readProblemNamesFromDir=false;
        for(String str:allProbs)
            problems.add(str);
        for(String cls:classifierList) {
            String parameters = "0";
            String[] str = {readPath,
                    "Z:\\ArchiveData\\Univariate_arff\\", folds + "", "false", cls, parameters};
            //Change this to read an array
            collate(str);
        }
    }


    public static void compareTopDogs() throws Exception {
        int folds=30;
        String readPath="E:/Results Working Area/Hybrids/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "top_dog_compare",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] newClassifiers={"TSCHIEF"};
        m.readInClassifiers(newClassifiers,readPath);
//        readPath="Z:/Results Working Area/DeepLearning/";
 //       newClassifiers=new String[] {"resnet","InceptionTime"};
 //       m.readInClassifiers(newClassifiers,readPath);
        readPath="E:/Results Working Area/HC Variants/";
        newClassifiers=new String[] {"HC-Catch22TSF"};
       m.readInClassifiers(newClassifiers,readPath);

        m.runComparison();


    }

    public static void C22IFPaperSection5_1MiniBakeoff() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/CIFPaper/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"Summary Results/",
                "miniBakeoff5_1_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] c={"catch22","DTWCV","HIVE-COTE","S-BOSS","PF","STC","TSF","TS-CHIEF","InceptionTime","WEASEL"};//,"HC-Latest","HC-CIF"};
        m.readInClassifiers(c,readPath+"Classifier Results/");
        m.runComparison();

    }


    public static void C22IFPaperSection5_2IntervalBased() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/CIFPaper/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"Summary Results/",
                "intervalbased5_2_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] c={"CIF","hybrid","TSF","catch22"};//,"HC-Latest","HC-CIF"};
        m.readInClassifiers(c,readPath+"Classifier Results/");
        m.runComparison();

    }



    public static void C22IFPaperSection5_2HCComponents() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"C22IFPaper/",
                "HC-components5_2_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] c={"CIF"};//,"HC-Latest","HC-CIF"};
        m.readInClassifiers(c,readPath+"IntervalBased/");
        String[] c2={"BOSS","RISE","TSF","EE","STC","HIVE-COTE"};
        m.readInClassifiers(c2,readPath+"HC Variants/");
        m.runComparison();

    }


    public static void C22IFPaperSection5_2NewComponents() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/CIFPaper/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"Summary Results/",
                "new-components5_2_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] c={"S-BOSS","STC","PF","WEASEL","CIF"};
        m.readInClassifiers(c,readPath+"Classifier Results/");
        m.runComparison();

    }


    public static void CIFPaperSection5_3SOTA() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/CIFPaper/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"Summary Results/",
                "Reboot_SOTA5_3_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] c={"HIVE-COTE","HC-CIF","TS-CHIEF","InceptionTime"};//,"HC-Latest","HC-CIF"};
        m.readInClassifiers(c,readPath+"Classifier Results/");
        m.runComparison();

    }

    public static void HC_SOTA() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/HC Variants/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "HC_tuning_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] c={"HIVE-COTE","TunedHIVE-COTE"};//,"HC-Latest","HC-CIF"};
        m.readInClassifiers(c,readPath);
        m.runComparison();

    }



    public static void C22IFPaperSectionSuppMaterialAll() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/CIFPaper/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"Summary Results/",
                "AllSuppMaterial_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] c={"catch22","DTWCV","CIF","HIVE-COTE","S-BOSS","PF","STC","TSF","TS-CHIEF","InceptionTime","WEASEL","HC-CIF"};//,"HC-Latest","HC-CIF"};
        m.readInClassifiers(c,readPath+"Classifier Results/");
        m.runComparison();

    }

    public static void tempCompare() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/FrequencyBased/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "RISE_TRAIN"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(false);
        String[] allClassifiersWithTestResults={"RISE-CVTRAIN","RISE-OOB"};
        m.readInClassifiers(allClassifiersWithTestResults,readPath);
        m.runComparison();

    }
    public static void tsfCompare() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/TSF Test/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "TSFResults_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(false);
        String[] allClassifiersWithTestResults={"TSFBagging","TSFCV_Full","TSFOOB_Full","TunedTSF"};
        m.readInClassifiers(allClassifiersWithTestResults,readPath);
        m.runComparison();

    }
    public static void TDEvsDictionary() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"TDEPaper/",
                "DictionaryComparison",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        allProbs=tscProblems107;
        m.setDatasets(allProbs);
//        m.setTestResultsOnly(true);
        String[] ts={"BOSS","cBOSS","WEASEL","S-BOSS","TDE","cS-BOSS"};
        m.readInClassifiers(ts,readPath+"DictionaryBased/");


        m.runComparison();

    }


    public static void HC_TDE_vsSOTA
            () throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"TempResults/",
                "SOTA",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
//        String[] allClassifiersWithTestResults={"HC-TDE2","HIVE-COTE2","HC-WEASEL2"};
 //       m.readInClassifiers(allClassifiersWithTestResults,readPath+"TDEPaper/");
        String[] str2={"HIVE-COTE"};//
        m.readInClassifiers(str2,readPath+"TDEPaper/");
        String[] ts={"TS-CHIEF"};
        m.readInClassifiers(ts,readPath+"Hybrids/");
        String[] incep={"InceptionTime","ROCKET"};
        m.readInClassifiers(incep,readPath+"DeepLearning/");
        m.runComparison();

    }

    public static void HC_Variants() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/TDEPaper/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "HC_Variants",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
//        String[] allClassifiersWithTestResults={"HC-TDE2","HIVE-COTE2","HC-WEASEL2"};
        //       m.readInClassifiers(allClassifiersWithTestResults,readPath+"TDEPaper/");
        String[] str2={"HC-TDE","HIVE-COTE", "HC-WEASEL","HC-S-BOSS"};
        m.readInClassifiers(str2,readPath);
        str2=new String[]{"InceptionTime","ROCKET"};
        m.readInClassifiers(str2,"E:/Results Working Area/DeepLearning/");
        str2=new String[]{"TS-CHIEF"};
        m.readInClassifiers(str2,"E:/Results Working Area/Hybrids/");

        m.runComparison();

    }



    public static void TDEContractCompare() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"TDEPaper/",
                "TDEContract_Folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
//        String[] allClassifiersWithTestResults={"HC-TDE2","HIVE-COTE2","HC-WEASEL2"};
        //       m.readInClassifiers(allClassifiersWithTestResults,readPath+"TDEPaper/");
        String[] str2={"TDE","TDE-1H"};//
        m.readInClassifiers(str2,readPath+"TDEPaper/");
        m.runComparison();

    }


    public static void componentTest() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"HCv2Paper/",
                "Component_TEST"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
//        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=tscProblems107;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(false);
        String[] g1={"BOSS","STC","RISE","TSF","BcS-BOSS"};
        m.readInClassifiers(g1,readPath+"cS-BOSSPaper/");
        String[] g2={"CIF","PF","RISEV2"};
        m.readInClassifiers(g2,readPath+"HCv2Paper/");
        m.runComparison();
    }
    public static void bestCompare() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"HCv2Paper/",
                "Hybrid_"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
//        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        String[] allClassifiersWithTestResults={"HC-TED2","HIVE-COTE2"};
        m.readInClassifiers(allClassifiersWithTestResults,readPath+"cS-BOSSPaper/");
        String[] ts={"TSCHIEF"};
        m.readInClassifiers(ts,readPath+"Hybrids/");
        String[] incep={"InceptionTime"};
        m.readInClassifiers(incep,readPath+"DeepLearning/");
        String[] hc={"HIVE-COTEV2","HC-V2NoRise"};
        m.readInClassifiers(hc,readPath+"HCv2Paper/");
        m.runComparison();

    }


    //Missing
    //<editor-fold defaultstate="collapsed" desc="tsc Problems 2018, no missing values">
    public static String[] tscProblems107={
            //Train Size, Test Size, Series Length, Nos Classes
            "ACSF1",
            "Adiac",        // 390,391,176,37
            "ArrowHead",    // 36,175,251,3
            "Beef",         // 30,30,470,5
            "BeetleFly",    // 20,20,512,2
            "BirdChicken",  // 20,20,512,2
            "BME",
            "Car",          // 60,60,577,4
            "CBF",                      // 30,900,128,3
            "Chinatown",
            "ChlorineConcentration",    // 467,3840,166,3
            "CinCECGTorso", // 40,1380,1639,4
            "Coffee", // 28,28,286,2
            "Computers", // 250,250,720,2
            "CricketX", // 390,390,300,12
            "CricketY", // 390,390,300,12
            "CricketZ", // 390,390,300,12
            "Crop",
            "DiatomSizeReduction", // 16,306,345,4
            "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
            "DistalPhalanxOutlineCorrect", // 600,276,80,2
            "DistalPhalanxTW", // 400,139,80,6
            "Earthquakes", // 322,139,512,2
            "ECG200",   //100, 100, 96
            "ECG5000",  //4500, 500,140
            "ECGFiveDays", // 23,861,136,2
            "EOGHorizontalSignal",
            "EOGVerticalSignal",
            "EthanolLevel",
            "FaceAll", // 560,1690,131,14
            "FaceFour", // 24,88,350,4
            "FacesUCR", // 200,2050,131,14
            "FiftyWords", // 450,455,270,50
            "Fish", // 175,175,463,7
            "FreezerRegularTrain",
            "FreezerSmallTrain",
            "GunPoint", // 50,150,150,2
            "GunPointAgeSpan",
            "GunPointMaleVersusFemale",
            "GunPointOldVersusYoung",
            "Ham",      //105,109,431
            "Haptics", // 155,308,1092,5
            "Herring", // 64,64,512,2
            "HouseTwenty",
            "InlineSkate", // 100,550,1882,7
            "InsectEPGRegularTrain",
            "InsectEPGSmallTrain",
            "InsectWingbeatSound",//1980,220,256
            "ItalyPowerDemand", // 67,1029,24,2
            "LargeKitchenAppliances", // 375,375,720,3
            "Lightning2", // 60,61,637,2
            "Lightning7", // 70,73,319,7
            "Mallat", // 55,2345,1024,8
            "Meat",//60,60,448
            "MedicalImages", // 381,760,99,10
            "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
            "MiddlePhalanxOutlineCorrect", // 600,291,80,2
            "MiddlePhalanxTW", // 399,154,80,6
            "MixedShapesRegularTrain",
            "MixedShapesSmallTrain",
            "MoteStrain", // 20,1252,84,2
            "OliveOil", // 30,30,570,4
            "OSULeaf", // 200,242,427,6
            "PhalangesOutlinesCorrect", // 1800,858,80,2
            "Phoneme",//1896,214, 1024
            "PigAirwayPressure",
            "PigArtPressure",
            "PigCVP",
            "Plane", // 105,105,144,7
            "PowerCons",
            "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
            "ProximalPhalanxOutlineCorrect", // 600,291,80,2
            "ProximalPhalanxTW", // 400,205,80,6
            "RefrigerationDevices", // 375,375,720,3
            "Rock",
            "ScreenType", // 375,375,720,3
            "SemgHandGenderCh2",
            "SemgHandMovementCh2",
            "SemgHandSubjectCh2",
            "ShapeletSim", // 20,180,500,2
            "ShapesAll", // 600,600,512,60
            "SmallKitchenAppliances", // 375,375,720,3
            "SmoothSubspace",
            "SonyAIBORobotSurface1", // 20,601,70,2
            "SonyAIBORobotSurface2", // 27,953,65,2
            "StarLightCurves", // 1000,8236,1024,3
            "Strawberry",//370,613,235
            "SwedishLeaf", // 500,625,128,15
            "Symbols", // 25,995,398,6
            "SyntheticControl", // 300,300,60,6
            "ToeSegmentation1", // 40,228,277,2
            "ToeSegmentation2", // 36,130,343,2
            "Trace", // 100,100,275,4
            "TwoLeadECG", // 23,1139,82,2
            "TwoPatterns", // 1000,4000,128,4
            "UMD",
            "UWaveGestureLibraryAll", // 896,3582,945,8
            "UWaveGestureLibraryX", // 896,3582,315,8
            "UWaveGestureLibraryY", // 896,3582,315,8
            "UWaveGestureLibraryZ", // 896,3582,315,8
            "Wafer", // 1000,6164,152,2
            "Wine",//54	57	234
            "WordSynonyms", // 267,638,270,25
            "Worms", //77, 181,900,5
            "WormsTwoClass",//77, 181,900,5
            "Yoga" // 300,3000,426,2
    };
    //</editor-fold>



    public static void memoryCompare() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/TDEPaper/MemoryCompare/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "MemoryComparison",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(false);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
//        String[] allClassifiersWithTestResults={"HC-TDE2","HIVE-COTE2","HC-WEASEL2"};
        //       m.readInClassifiers(allClassifiersWithTestResults,readPath+"TDEPaper/");
        String[] str2={"BOSS","S-BOSS", "WEASEL"};
        m.readInClassifiers(str2,readPath);
        str2=new String[]{"cBOSS","cS-BOSS","TDE"};
        m.readInClassifiers(str2,"Z:/Results Working Area/DictionaryBased/");

        m.runComparison();

    }


    public static void HC_Components() throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/HIVE-COTE 1.0/";
        String[] ts={"TSF","RISE","cBOSS","STC","HC 1.0"};
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "HC-Analysis",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        m.readInClassifiers(ts,readPath);
        m.runComparison();
  }

    public static void HC_vsSOTA
            () throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"HIVE-COTE 1.0/",
                "HC-vs-SOTA_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
//        String[] allClassifiersWithTestResults={"HC-TDE2","HIVE-COTE2","HC-WEASEL2"};
        //       m.readInClassifiers(allClassifiersWithTestResults,readPath+"TDEPaper/");
        String[] str2={"HC 1.0"};//
        m.readInClassifiers(str2,readPath+"HIVE-COTE 1.0/");
        String[] ts={"TS-CHIEF"};
        m.readInClassifiers(ts,readPath+"Hybrids/");
        String[] incep={"InceptionTime","ROCKET"};
        m.readInClassifiers(incep,readPath+"DeepLearning/");
        m.runComparison();

    }
    public static void HC_All
            () throws Exception {
        int folds = 30;
        String readPath = "E:/Results Working Area/";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath+"HIVE-COTE 1.0/",
                "HC-all_folds"+folds,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
//        String[] allClassifiersWithTestResults={"HC-TDE2","HIVE-COTE2","HC-WEASEL2"};
        //       m.readInClassifiers(allClassifiersWithTestResults,readPath+"TDEPaper/");
        String[] str2={"TSF","RISE","cBOSS","STC","HC 1.0"};//
        m.readInClassifiers(str2,readPath+"HIVE-COTE 1.0/");
        String[] ts={"TS-CHIEF"};
        m.readInClassifiers(ts,readPath+"Hybrids/");
        String[] incep={"InceptionTime","ROCKET"};
        m.readInClassifiers(incep,readPath+"DeepLearning/");
        m.runComparison();

    }

    public static void temp() throws Exception {
        int folds = 1;
        String readPath = "E:/Results Working Area/HIVE-COTE 1.0/";
        String[] ts={"HIVE-COTE","TSCHIEF"};
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "temp",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        m.readInClassifiers(ts,readPath);
        m.runComparison();
    }
    public static void singleProblem(String classifier) throws Exception {
        int folds = 30;
        String readPath = "Z:/ReferenceResults/";
        String[] ts={classifier};
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                classifier+"/SummaryEvaluation",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        m.readInClassifiers(ts,readPath);
        m.runComparison();
    }

    public static void makeMegaCD() throws Exception {
        int folds = 30;
        String readPath = "Z:/ReferenceResults/";
        String[] ts={"BOSS","Catch22","cBOSS","HIVE-COTE v1.0","InceptionTime","ProximityForest","ResNet",
                    "RISE", "ROCKET","S-BOSS","STC","TS-CHIEF","TSF","WEASEL"};
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                "MegaComparison",folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs=DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        m.readInClassifiers(ts,readPath);
        m.runComparison();

    }




    public static void main(String[] args) throws Exception {
        //TDE vs BOSS, cBOSS, cS-BOSS, WEASEL and S-BOSS
//        TDEvsDictionary();
        //HC-TDE vs HIVE-COTE, TS-CHIEF and InceptionTime
 //       HC_Components();
  //      HC_All();
 //       HC_vsSOTA();
        //HC-TDE vs HC-S-BOSS and HC-WEASEL
 //       memoryCompare();
//        HC_Variants();

 //       tempSummary();
  //      temp();
 //       makeMegaCD();
  //      System.exit(0);
        String type="";
        String[] datasets=DatasetLists.fixedLengthMultivariate;
        type="QuickStats";
       type="SingleClassifiers";
  //     type="MultipleClassifiers";
        if (args.length == 0) {//Local run: manually configure

            int folds=30;

            String readPath="E:\\Results Working Area\\Multivariate\\CompleteClassifiers\\";
//            String readPath="Z:/Results and Code for Papers/CIF-KDD2020/Classifier Results/";
            // String[] classifierList={"BOSS","cBOSS","S-BOSS","WEASEL","cS-BOSS","TDE"};
            String[] classifierList={"CBOSS_I","TSF_I","RISE_I","STC_I","HIVE-COTE_I"};
//            String readPath="E:/Results Working Area/DeepLearning/";
//            singleProblem(classifierList[0]);
            switch(type){
                case "QuickStats": 
//OR DIRECTLY IF YOU WANT
//quickStats("C:/Users/ajb/Dropbox/results david/ShapeletForest",false,1,"HIVE-COTE,ST","HIVE-COTE,BOSS","HIVE-COTE,HIVE-COTE");
//          quickStats("C:/Users/ajb/Dropbox/results david/CNN100hours",false,1);
  //PYTHON VERSIONS
  //         quickStats("E:/Results/sktimeResults/TSF",false,30,"HIVE-COTE,TSF");
  //         quickStats("E:/Results/sktimeResults/BOSS",false,30,"HIVE-COTE,BOSS");
  //         quickStats("E:/Results/sktimeResults/RISE",false,30,"HIVE-COTE,RISE");
                    boolean recalAcc=false;
                    //These for quick stats
                    bakeOffPath=bakeOffPathBeast;
                    hiveCotePath=hiveCotePathBeast;
                    reduxPath= reduxPathBeast;
                    for(String cls:classifierList){
    //                    String comparisons=null;
                        String[] comparisons={"HIVE-COTE,TSF","HIVE-COTE,RISE","HIVE-COTE,BOSS","HIVE-COTE,ST","HIVE-COTE,HIVE-COTE"};
                        args=makeQuickStatsArgs(readPath+cls,recalAcc,folds,comparisons);
                        singleClassifiervsReferenceResults(args);
                    }
                break;
                case "SingleClassifiers":
                    problems=new ArrayList<>();
                    readProblemNamesFromDir=false;
                    for(String str:datasets)
                        problems.add(str);
                    for(String cls:classifierList){
                        System.out.println(" Processing "+cls);
                        String parameters="0";
                        String[] str={readPath,
                            "Z:\\ArchiveData\\Univariate_arff\\",folds+"","false",cls,parameters};
            //Change this to read an array            
                        collate(str);
                    }
                    break;
                case "MultipleClassifiers":
//Where to put them, directory name, number of folds
                    String[] name=readPath.split("/");
                    MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath, name[name.length-1]+"_"+folds+"_Resamples", folds);
                    m.setIgnoreMissingResults(true);
                    m.setBuildMatlabDiagrams(true);
                    m.setDebugPrinting(true);
                    m.setUseAllStatistics();
                    m.setDatasets(datasets);

//                    m.setDatasets(reduxComplete);
                    m.setTestResultsOnly(true);
                    m.readInClassifiers(classifierList,readPath);
                    m.runComparison();
                    
                    
                    
                break;

            }
        }
        else{           //Cluster run
            bakeOffPath=bakeOffPathCluster;
            hiveCotePath=hiveCotePathCluster;
            System.out.println("Cluster Job Args:");
            for(String s:args)
                System.out.println(s);
            switch(type){
                case "QuickStats":
                    singleClassifiervsReferenceResults(args);
                    break;
                case "SingleClassifiers":
                    singleClassifiersFullStats(args);
                    break;
                case "MultipleClassifier":
                    multipleClassifierFullStats(args);    
                default:
                    System.out.println("Unknown type = "+type);
            }
        }
    }

    
    
    
}
