/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package experiments;

import java.io.File;
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
                ClassificationExperiments.main(experimentArguments);
            }
            scan.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
