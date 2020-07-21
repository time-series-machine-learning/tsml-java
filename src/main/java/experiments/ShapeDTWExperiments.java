package experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class ShapeDTWExperiments {
    public static void main(String[] args){
        try {
            String fileLoc = "C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\datasetsList.txt";
            Scanner scan = new Scanner(new File(fileLoc));
            while(scan.hasNextLine()) {
                String [] experimentArguments = new String[5];
                experimentArguments[0] = "--dataPath=C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_arff";
                experimentArguments[1] = "--resultsPath=C:\\Users\\Vince\\Documents\\Dissertation Repositories\\results\\java";
                experimentArguments[2] = "--classifierName=NN_ShapeDTW_Raw";
                experimentArguments[3] = "--datasetName=" + scan.nextLine();
                experimentArguments[4] = "--fold=10";
                Experiments.main(experimentArguments);
            }
            scan.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
