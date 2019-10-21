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
    static boolean oldFormat=true;
  
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
   public static void singleClassifierFullStats(String[] args) throws Exception{
        if(args.length>1)
            collate(args);
        else{ 
        String[] classifiers={"TSF"};
        for(String classifier:classifiers){
            String parameters="0";
            String[] str={"E:\\Results\\Java\\",
                "Z:\\Data\\TSCProblems2018\\","30","false",classifier,parameters};
            collate(str);
        }
    }

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
   public static String bakeOffPath="Z:/ReferenceResults/CollatedResults/Bakeoff2015/byClassifier/";
   public static String hiveCotePath="Z:/ReferenceResults/CollatedResults/HIVE-COTE2017/";
   public static String bakeOffPathBeast="Z:/ReferenceResults/CollatedResults/Bakeoff2015/byClassifier/";
   public static String hiveCotePathBeast="Z:/ReferenceResults/CollatedResults/HIVE-COTE2017/";
   public static String reduxPathBeast="Z:/ReferenceResults/CollatedResults/BakeoffRedux2019/";
   public static String bakeOffPathCluster="/gpfs/home/ajb/ReferenceResults/Bakeoff2015/ByClassifier/";
   public static String hiveCotePathCluster="/gpfs/home/ajb/ReferenceResults/HIVE-COTE2017/";
   public static String reduxPathCluster="/gpfs/home/ajb/ReferenceResults/BakeoffRedux2019/";

   
   
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
        folds=Integer.parseInt(args[2]);
        oldFormat=Boolean.parseBoolean(args[3]);
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
    public static void quickStats(String primary, boolean calcAcc, int folds, boolean oldForm,String...others) throws Exception{
        String[] input;
        if(others==null)
            input=new String[4];
        else 
            input=new String[4+others.length];
        input[0]=primary;
        input[1]=calcAcc+"";
        input[2]=folds+"";
        input[3]=oldForm+"";
        if(others!=null)
            for(int i=0;i<others.length;i++)
                input[i+4]=others[i];
       singleClassifiervsReferenceResults(input);
        
    }
    public static void main(String[] args) throws Exception {
 
        
        
        if (args.length == 0) {//Local run
            bakeOffPath=bakeOffPathBeast;
            hiveCotePath=hiveCotePathBeast;
            quickStats("C:/Temp/CNN/CNN10hours",false,1,false);
            //,"HIVE-COTE,ST","HIVE-COTE,HIVE-COTE");
//          quickStats("Z:/Results/BOSS variants/Univariate/RBOSS250",false,30,false,"HIVE-COTE,BOSS");
//TunedTSF
 //           quickStats("E:/Results/UCR Debug/Java/TunedTSF",false,30,"Bakeoff,ST","Bakeoff,TSF","Bakeoff,BOSS","Bakeoff,DTWCV");
 //ProximityForest
//            quickStats("E:/Results/UCR Debug/Java/ProximityForest",false,30,"HIVE-COTE,EE","HIVE-COTE,BOSS","HIVE-COTE,TSF","HIVE-COTE,RISE","HIVE-COTE,ST","HIVE-COTE,HIVE-COTE");

 //           quickStats("Z:/Results/Post Bakeoff Results/resnet/",false,30,"HIVE-COTE,EE","HIVE-COTE,BOSS","HIVE-COTE,TSF","HIVE-COTE,RISE","HIVE-COTE,ST","HIVE-COTE,HIVE-COTE");
  //         quickStats("Z:/Results/Post Bakeoff Results/WEASEL/",false,30,"HIVE-COTE,EE","HIVE-COTE,BOSS","HIVE-COTE,TSF","HIVE-COTE,RISE","HIVE-COTE,ST","HIVE-COTE,HIVE-COTE");
             

//REDUX: EE
 //           quickStats("Z:/Results/Bakeoff Redux/Java/EE",false,30,"HIVE-COTE,EE","Bakeoff,EE");
 //REDUX: TSF
//            quickStats("Z:/Results/Bakeoff Redux/Java/TSF",false,30,"HIVE-COTE,TSF","Bakeoff,TSF");
 //REDUX: BOSS
//            quickStats("Z:/Results/Bakeoff Redux/Java/BOSS",false,30,"HIVE-COTE,BOSS","Bakeoff,BOSS");
 //REDUX: RISE
  
//            quickStats("E:/Results/UCR Debug/Python/TSF",false,30,"HIVE-COTE,TSF","HIVE-COTE,EE","HIVE-COTE,BOSS","HIVE-COTE,RISE","HIVE-COTE,ST","HIVE-COTE,HIVE-COTE");
 //           quickStats("Z:/Results/Bakeoff Redux/Java/RISE",false,30,"HIVE-COTE,RISE");
 //REDUX: ST
//            quickStats("Z:/Results/Bakeoff Redux/Java/ST",false,30,"HIVE-COTE,ST","Bakeoff,ST");
///REDUX: HIVE-COTE
//           quickStats("Z:/Results/Bakeoff Redux/Java/HIVE-COTE",false,30,"HIVE-COTE,HIVE-COTE");

            
            
        }
        else{           //Cluster run
            bakeOffPath=bakeOffPathCluster;
            hiveCotePath=hiveCotePathCluster;
            System.out.println("Cluster Job Args:");
            for(String s:args)
                System.out.println(s);
            singleClassifiervsReferenceResults(args);
        }
            
        System.exit(0);
        boolean singleClassifierStats=true;
        if(singleClassifierStats)
            singleClassifierFullStats(args);
        else
            multipleClassifierFullStats(args);
    } 
    
    
    
    
    
    
}
