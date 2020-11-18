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
package evaluation;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.OutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import statistics.tests.OneSampleTests;
import statistics.tests.OneSampleTests;
import statistics.tests.TwoSampleTests;

/**
 * Reads in a file of accuracies for k classifiers and generates a kxk matrix
 * of p-values. INPUT FORMAT:
          ,Classifier1,Classifier2,Classifier3, ...,Classifierk
 Problem1 , 0.5,....
 Problem2 , 0.5,....
 .
 .
 ProblemN, 0.5,....
 
 Output: Pairwise matrix of difference and a version of results.
 * 
 * @author ajb
 */
public class MultipleClassifiersPairwiseTest {
    public static boolean beQuiet = false;
    
    static double[][] accs; //ROW indicates classifier, for ease of processing
    static double[][] pValsTTest; //ROW indicates classifier, for ease of processing
    static double[][] pValsSignTest; //ROW indicates classifier, for ease of processing
    static double[][] pValsSignRankTest; //ROW indicates classifier, for ease of processing
    static double[][] bonferonni_pVals; //ROW indicates classifier, for ease of processing
    static boolean[][] noDifference; //ROW indicates classifier, for ease of processing
    
    static int nosClassifiers;
    static int nosProblems;
    static String[] names;
    
/** Assumes classifier names in the first line and problem names in the first column
 */ public static void loadData(String file, PrintStream out){
        InFile data=new InFile(file);
        nosProblems=data.countLines()-1;
        data=new InFile(file);
        String[] temp=data.readLine().split(",");
        nosClassifiers=temp.length-1;
        names=new String[nosClassifiers];
        for(int i=0;i<nosClassifiers;i++)
            names[i]=temp[i+1];
        accs=new double[nosClassifiers][nosProblems];
        for(int j=0;j<nosProblems;j++){
            String[] line = data.readLine().split(",");
            if(!beQuiet)
                System.out.print("Problem ="+line[0]+",");
            for(int i=0;i<nosClassifiers;i++){
                accs[i][j]=Double.parseDouble(line[i+1]);
                if (!beQuiet)
                    out.print(accs[i][j]+",");
            }
            if (!beQuiet)
                out.print("\n");
            
        }
    }
    
    public static void loadData(String file){
        loadData(file, System.out);
    }
    
    public static void findPVals(){
        pValsTTest=new double[nosClassifiers][nosClassifiers];
        pValsSignTest=new double[nosClassifiers][nosClassifiers];
        pValsSignRankTest=new double[nosClassifiers][nosClassifiers];
        OneSampleTests test=new OneSampleTests();
        for(int i=0;i<nosClassifiers;i++)
        {
            for(int j=i+1;j<nosClassifiers;j++){
//Find differences
                double[] diff=new double[accs[i].length];
                for(int k=0;k<accs[i].length;k++)
                    diff[k]=accs[i][k]-accs[j][k];
                String str=test.performTests(diff);
                
                if(!beQuiet)
                    System.out.println("TEST Classifier "+names[i]+" VS "+names[j]+ " returns string "+str);
                
                String[] tmp=str.split(",");
                pValsTTest[i][j]=Double.parseDouble(tmp[2]);
                pValsSignTest[i][j]=Double.parseDouble(tmp[5]);
                pValsSignRankTest[i][j]=Double.parseDouble(tmp[8]);
            }
        }
    }
    public static void findMeanDifferences(String file){
        double[][] meanDiff=new double[nosClassifiers][nosClassifiers];
        OutFile outf=new OutFile(file);
        for(int i=0;i<nosClassifiers;i++)
        {
            for(int j=i+1;j<nosClassifiers;j++){
                for(int k=0;k<accs[i].length;k++)
                    meanDiff[i][j]+=accs[i][k]-accs[j][k];
                meanDiff[i][j]/=accs[i].length;
                meanDiff[j][i]=-meanDiff[i][j];
//                meanDiff[i][j]*=-1;
            }
        }
        for(int i=0;i<nosClassifiers;i++)
        {
            for(int j=0;j<nosClassifiers;j++)
                outf.writeString(meanDiff[i][j]+",");
            outf.writeString("\n");
        }
    }

    /**
     *
     * @param alpha
     * @param printPVals
     */
    public static void findDifferences(double alpha,boolean printPVals){

        noDifference=new boolean[nosClassifiers][nosClassifiers];
        for(int i=0;i<nosClassifiers;i++)
        {
            noDifference[i][i]=true;
/*            for(int j=nosClassifiers-1;j>i;j--) {
                int numComparisons = nosClassifiers - i;
                noDifference[i][j] = true;
                noDifference[j][i] = true;
                double criticalValue = alpha / (numComparisons - (j - nosClassifiers - 1));
//                System.out.println(" Critical value = "+criticalValue);
                if (pValsSignRankTest[i][j] < criticalValue) {
                    noDifference[i][j] = false;
                }
            }
*/
            for(int j=i+1;j<nosClassifiers;j++){
                noDifference[i][j]=true;
                noDifference[j][i]=true;
                if(pValsSignRankTest[i][j]<alpha){
//                if(pValsTTest[i][j]<alpha && pValsSignTest[i][j]< alpha && pValsSignRankTest[i][j]<alpha)
                    noDifference[i][j]=false;
                    noDifference[j][i]=false;
                }
            }
        }
        DecimalFormat df = new DecimalFormat("##.#####");

    }

    /**
     * This has a built in Holm correction
      * @param input
     * @param output
     */
    public static void runTests(String input, String output){
         loadData(input);
//        loadData("C:\\Research\\Papers\\2016\\JMLR HIVE-COTE Jason\\RiseTestWithNames.csv");
        findPVals();
        double alpha=0.1;
//printPVals=false;
//Bonferonni adjusted        
//        alpha/=nosClassifiers*(nosClassifiers-1)/2;
//Control adjusted 
        alpha/=nosClassifiers-1;
        findDifferences(alpha,true);
        //Sort classifiers by rank: assume already done
        OutFile cliques=new OutFile(output);
        for(int i=0;i<nosClassifiers;i++){
            for(int j=0;j<nosClassifiers;j++)
                cliques.writeString(noDifference[i][j]+",");
            cliques.writeString("\n");
        }
       
    }

    public static StringBuilder runTests(double[][] d,String[] n){
        nosProblems=d.length;
        nosClassifiers=d[0].length;
        names=n;
        accs=d;
        findPVals();
        double alpha=0.05;
//printPVals=false;
//Bonferonni adjusted        
//        alpha/=nosClassifiers*(nosClassifiers-1)/2;
//Control adjusted 
        alpha/=nosClassifiers-1;
        findDifferences(alpha,true);
        StringBuilder results=new StringBuilder();
        
        results.append("T TEST");
        for(int i=0;i<nosClassifiers;i++)
            results.append(",").append(names[i]);
        results.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            results.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                results.append(",").append(pValsTTest[i][j]);
            results.append("\n");
        }
        results.append("\n");
        
        results.append("SIGN TEST");
        for(int i=0;i<nosClassifiers;i++)
            results.append(",").append(names[i]);
        results.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            results.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                results.append(",").append(pValsSignTest[i][j]);
            results.append("\n");
        }
        results.append("\n");
        
        results.append("SIGN RANK TEST");
        for(int i=0;i<nosClassifiers;i++)
            results.append(",").append(names[i]);
        results.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            results.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                results.append(",").append(pValsSignRankTest[i][j]);
            results.append("\n");
        }
        results.append("\n");
        
        results.append("NOSIGDIFFERENCE");
        for(int i=0;i<nosClassifiers;i++)
            results.append(",").append(names[i]);
        results.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            results.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                results.append(",").append(noDifference[i][j]);
            results.append("\n");
        }
        
        return results;
    }
    
    
 
    public static StringBuilder runSignRankTest(double[][] d,String[] n){
        nosProblems=d.length;
        nosClassifiers=d[0].length;
        names=n;
        accs=d;
        findPVals();
        double alpha=0.05;
//printPVals=false;
//Bonferonni adjusted        
//        alpha/=nosClassifiers*(nosClassifiers-1)/2;
//Control adjusted 
        alpha/=nosClassifiers-1;
        findDifferences(alpha,true);
        StringBuilder results=new StringBuilder();
        results.append("SIGN RANK TEST \n ");
        for(int i=0;i<nosClassifiers;i++)
            results.append(",").append(names[i]);
        results.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            results.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                results.append(",").append(pValsSignRankTest[i][j]);
            results.append("\n");
        }
        return results;
    }

    /**
     *
      * @param input
     * @return
     */
   
    public static StringBuilder runTests(String input){
         loadData(input);
        findPVals();
        double alpha=0.05;
//printPVals=false;
//Bonferonni adjusted        
//        alpha/=nosClassifiers*(nosClassifiers-1)/2;
//Control adjusted 
//        alpha/=nosClassifiers-1;
        findDifferences(alpha,true);
        
        StringBuilder cliques=new StringBuilder();
        
        cliques.append("T TEST");
        for(int i=0;i<nosClassifiers;i++)
            cliques.append(",").append(names[i]);
        cliques.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            cliques.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                cliques.append(",").append(pValsTTest[i][j]);
            cliques.append("\n");
        }
        cliques.append("\n");
        
        cliques.append("SIGN TEST");
        for(int i=0;i<nosClassifiers;i++)
            cliques.append(",").append(names[i]);
        cliques.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            cliques.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                cliques.append(",").append(pValsSignTest[i][j]);
            cliques.append("\n");
        }
        cliques.append("\n");
        
        cliques.append("SIGN RANK TEST");
        for(int i=0;i<nosClassifiers;i++)
            cliques.append(",").append(names[i]);
        cliques.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            cliques.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                cliques.append(",").append(pValsSignRankTest[i][j]);
            cliques.append("\n");
        }
        cliques.append("\n");
        
        cliques.append("NOSIGDIFFERENCE");
        for(int i=0;i<nosClassifiers;i++)
            cliques.append(",").append(names[i]);
        cliques.append("\n");
        for(int i=0;i<nosClassifiers;i++){
            cliques.append(names[i]);
            for(int j=0;j<nosClassifiers;j++)
                cliques.append(",").append(noDifference[i][j]);
            cliques.append("\n");
        }
        
        return cliques;
    }
    
    public static String printCliques() {
        StringBuilder sb = new StringBuilder();

        sb.append("cliques = [");
        boolean[][] cliques = findCliques(noDifference);
        for (int i = 0; i < cliques.length; i++) {
            for (int j = 0; j < cliques[i].length; j++)
                sb.append(cliques[i][j] ? "1" : 0).append(" ");
            sb.append("\n");
        }
        sb.append("]\n");

        return sb.toString();
    }

    public static boolean[][] findCliques(boolean[][] same) {
        //want to find the largest non-contained 'blocks' in the matrix where all values are true, i.e. all classifiers
        //are similar to all over classifiers within the block.
        //
        //if 1 = true, 0 = false, for input data
        // 1 1 1 0 0
        // 1 1 1 1 0
        // 1 1 1 1 0
        // 0 1 1 1 1
        // 0 0 0 1 1
        //
        //the resulting cliques would be
        // 1 1 1 0 0
        // 0 1 1 1 0
        // 0 0 0 1 1

        //starting each search on the diagonal (where similar always = true), shall try to grow a square down and to
        //the right where all elements are true. stop when a false would be included in the square. if the square is
        //at least size 2 already, this is a clique.

        List<List<Integer>> cliques = new ArrayList<>();
        int prevEndOfClique = 0;
        for (int i = 0; i < same.length; i++) {
            List<Integer> clique = new ArrayList<>(Arrays.asList(i));
            growClique(same, clique);

            if (clique.size() > 1) {
                //potential new clique, check it's not contained within the previous, i.e. a new classifier has been
                //included at the end of the clique
                int endOfClique = clique.get(clique.size()-1);
                if (endOfClique > prevEndOfClique) {
                    cliques.add(clique);
                    prevEndOfClique = endOfClique;
                }
                //else is just a subclique of the previous
            }

        }

        boolean[][] finalCliques = new boolean[cliques.size()][same.length];
        for (int i = 0; i < cliques.size(); ++i) {
            for (int j = 0; j < cliques.get(i).size(); ++j) {
                finalCliques[i][cliques.get(i).get(j)] = true;
            }
        }
        return finalCliques;
    }

    private static void growClique(boolean[][] same, List<Integer> clique) {
        int prevVal = clique.get(clique.size()-1);
        if (prevVal == same.length-1)
            return; // reached the end of the classifiers, no more room to grow


        int cliqueStart = clique.get(0);
        int nextVal = prevVal+1;

        // suppose size of clique is already 2, we'll need to check that all of the 1's are to true to grow the clique
        // 0 0 1
        // 0 0 1
        // 1 1 1
        // but actually, it should be mirrored about the diagonal, and the diagonal itself is always true, so really only need to check
        // 0 0 0      0 0 1
        // 0 0 0  or  0 0 1
        // 1 1 0      0 0 0

        for (int col = cliqueStart; col < nextVal; col++) {
            if (!same[nextVal][col]) {
                //found that growing the clique would include a false, quit now
                return;
            }
        }

        //all checks passed, add on this index and try to grow again
        clique.add(nextVal);
        growClique(same, clique);
    }

    private static void testNewCliques() {
        boolean[][] same = { //dtw example from teams
                { true,  true,  true,  false, false, false, },
                { true,  true,  true,  true,  false, false, },
                { true,  true,  true,  false, true,  true, },
                { false, true,  false, true,  true,  true, },
                { false, false, true,  true,  true,  true, },
                { false, false, true,  true,  true,  true, },
        };

        noDifference = same;

        System.out.println(printCliques());
    }



    //jamesl note: old clique forming used from roughly late 2017 to august2020 produced contained cliques/edge cases
    //in certain situations that were fixed by hand. now taking the clique forming approach of strictly all self similar
    //within blocks. see new findCliques()
//    public static boolean[][] findCliques(boolean[][] same) {
//        boolean[][] cliques = new boolean[same.length][];
//        for (int i = 0; i < same.length; i++) {
//
//            boolean[] clique = new boolean[same.length];
//            boolean inClique = false;
//
//            for (int j = i+1; j < same[i].length; j++) {
//                //all before i assumed false, wrong side of the diagonal
//                //i,j assumed true, no sig diff with self
//                //however only set later, to take advantaged of a binary flag
//                //for the existance of a clique for this classifier
//                inClique = inClique || same[i][j];
//                clique[j] = same[i][j];
//            }
//            clique[i] = true; //self similarity always true
//
//            if (inClique) //if similarity with at least one other
//                if (!isSubClique(cliques, clique)) //if similarity not already represented within a previously found clique
//                    addClique(cliques, clique);
//        }
//
//        int numNull = 0;
//        for (int i = cliques.length-1; i >= 0; i--) {
//            if (cliques[i] == null)
//                numNull++;
//            else
//                break;
//        }
//
//        //shittiest way to avoid arraylists
//        boolean[][] finalCliques = new boolean[cliques.length-numNull][];
//        System.arraycopy(cliques, 0, finalCliques, 0, finalCliques.length);
//
//        return finalCliques;
//    }
//
//    public static void addClique(boolean[][] cliques, boolean[] newClique) {
//        for (int i = 0; i < cliques.length; i++) {
//            if (cliques[i] == null) {
//                cliques[i] = newClique;
//                break;
//            }
//        }
//    }
//
//    public static boolean isSubClique(boolean[][] cliques, boolean[] newClique) {
//        for (int i = 0; i < cliques.length && cliques[i]!=null; i++) {
//            boolean subOfThisClique = true;
//            for (int j = 0; j < cliques[i].length; j++) {
//                if (newClique[j] && !cliques[i][j]) //if j is similar in new, but is not similar in old, found a difference and therefore new is not sub of this one
//                    subOfThisClique = false;
//            }
//            if (subOfThisClique)
//                return true;
//        }
//        return false;
//    }
    
    public static void main(String[] args) {
        testNewCliques();

////ASSUME INPUT IN RANK ORDER, WITH TOP RANKED CLASSIFIER FIRST, WORST LAST
////        String input="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\BasicExperiments\\";
////        String output="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\BasicExperiments\\";
////        String[] allSimulators={"WholeSeriesElastic","Interval","Shapelet","Dictionary","ARMA","All"};
////        for(String s:allSimulators)
//        String input="C:\\Users\\ajb\\Dropbox\\For Eamonn\\MPvsBenchmark.csv";
//
//        input="C:\\Users\\ajb\\Dropbox\\Working docs\\Research\\RotF Paper\\Results Standard RotF\\Shapelet Results\\Results.csv";
////        String input="C:\\Research\\Results\\RepoResults\\HIVE Results";
//////
////        input="C:\\Research\\Papers\\2017\\PKDD BOP to BOSS\\Results\\vsCNN";
////        input="C:\\Users\\ajb\\Dropbox\\Temp\\test";
////        String s= "All";
////            runTests(input+s+"CombinedResults.csv",input+s+"Tests.csv");
////            runTests(input+".csv",input+"Tests.csv");
//            System.out.println(runTests(input).toString());
//            System.out.println("\n\n" + printCliques());
////findMeanDifferences(input+" MeanDiffs.csv");
    }
    
    
    
/* private static void createTable(File file, PrintStream out) throws FileNotFoundException{
    Scanner sc = new Scanner(file);
    
    while(sc.hasNextLine()){
        String[] data = sc.nextLine().split((","));
        
        String dataSet = data[0];
        float[] results = new float[data.length-1];
        int index=0;
        out.print(dataSet);
        for(int i=1; i<data.length; i++){
            results[i-1] =  Float.parseFloat(data[i]);                
            if(results[i-1] > results[index] ){
                index = i-1;
            }
        }
        
        for(int i=0; i<results.length; i++){
            String format = " & %s";
            if(index == i)
                format = " & {\\bf %f}";
            out.printf(format, results[i]);
        }
        
        out.printf("\\\\\n");
    }
    
}   
*/}
