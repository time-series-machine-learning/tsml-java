package ml_6002b_coursework;
import scala.tools.nsc.JarRunner;

import java.lang.Math;
import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    public static int[][] transpose(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        int[][] transposedMatrix = new int[n][m];

        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m; y++) {
                transposedMatrix[x][y] = matrix[y][x];
            }
        }

        return transposedMatrix;
    }

    public static void main(String[] args) {
        /**
         * Test harness for AttributeMeasures methods
         */
        int[][] contingencyTable = new int[][]{
                {4, 0},
                {1, 5},
        };

        System.out.println("measure Information Gain for Peaty = "+measureInformationGain(contingencyTable));
        System.out.println("measure Information Gain Ratio for Peaty = "+measureInformationGainRatio(contingencyTable));
        System.out.println("measure Gini for Peaty = "+measureGini(contingencyTable));
        System.out.println("measure Chi Squared  for Peaty = "+measureChiSquared(contingencyTable));


    }

    /**
     * Measures information gain of a contingency table
     * @param contingencyTable
     * @return Information Gain
     */
    public static double measureInformationGain(int[][] contingencyTable) {
        List<Double> rowTotal = new ArrayList<>();
        List<Double> columnTotal = new ArrayList<>();

        double count;
        for (int i = 0; i < contingencyTable.length; i++) {
            count = 0;

            for (int j = 0; j < contingencyTable[i].length; j++) {
                count = count + contingencyTable[i][j];
            }

            rowTotal.add(count);
        }


        int[][] transposedContingencyTable;
        transposedContingencyTable = transpose(contingencyTable);
        for (int i = 0; i < transposedContingencyTable.length; i++) {
            count = 0;

            for (int j = 0; j < transposedContingencyTable[i].length; j++) {
                count = count + transposedContingencyTable[i][j];
            }

            columnTotal.add(count);
        }




        double len = 0;
        for (int i = 0; i < rowTotal.size(); i++) {
            len = len + rowTotal.get(i);
        }

        double infGain = 0;
        double entropy;
        List<Double> chance = new ArrayList<>();
        double denominator = 0;

        //calculates the total datapoints provided
        for (int i = 0; i < columnTotal.size(); i++) {
            denominator += columnTotal.get(i);
        }

        /**
         * Entropy calculation proivided ML Lecture 2, page 25 and 26
         **/

        /**
         * chance iterates through all the potential outcomes and turns them into fractions based on the total
         */
        for (int i = 0; i < columnTotal.size(); i++) {
            if (denominator == 0) {
                return 1;
            } else {
                double temp = columnTotal.get(i) / denominator;
                chance.add(temp);
            }
        }

        /** NOTE
         * All logorithms in this class are log base 2 :
         * The conversion is done using the Math.log giving natural log then dividing it by log 2
         */
        entropy = 0;

        /**
         * in test case: -(0.5log(0.5) + 0.5log(0.5) = 1
         */
        for (int i = 0; i < columnTotal.size(); i++) {
            entropy += chance.get(i) * (Math.log(chance.get(i)) / Math.log(2));
        }
        entropy = -entropy;

        List<Double> Hx = new ArrayList<>();
        Hx.add(entropy);

        for (int i = 0; i < contingencyTable.length; i++) {
            double tempHx = 0;
            for (int j = 0; j < contingencyTable[i].length; j++) {
                if (contingencyTable[i][j] == 0.0) {
                    tempHx = 0.0;
                } else {
                    tempHx = tempHx + (contingencyTable[i][j] / rowTotal.get(i)) * ((Math.log(contingencyTable[i][j] / rowTotal.get(i))) / Math.log(2));
                }

            }
            tempHx = Math.abs(tempHx);
            Hx.add(tempHx);
        }

        infGain = Hx.get(0);
            for (int j = 0; j < rowTotal.size(); j++) {
                infGain -= (rowTotal.get(j) / denominator) * Hx.get(j + 1);
            }
        return infGain;
    }


    /**
     * Measures information gain ratio of a continggency table
     * takes information gain
     * Calculates split info
     * Calculates information gain ratio
     * @param contingencyTable
     * @return Information Gain Ratio
     */
    public static double measureInformationGainRatio(int[][] contingencyTable) {

        double infGain = measureInformationGain(contingencyTable);

        double splitInfo = 0;

        List<Double> rowTotal = new ArrayList<>();
        double count;

        for (int i = 0; i < contingencyTable.length; i++) {
            count = 0;
            for (int j = 0; j < contingencyTable[i].length; j++) {
                count = count + contingencyTable[i][j];
            }
            rowTotal.add(count);
        }
        /**
         * calculates the total datapoints provided
         */
        double denominator = 0;
        for (int i = 0; i < rowTotal.size(); i++) {
            denominator += rowTotal.get(i);
        }
        for (int i = 0; i < rowTotal.size(); i++) {
            splitInfo += (rowTotal.get(i) / denominator) * (Math.log(rowTotal.get(i) / denominator) / Math.log(2));

        }

        double infGainRatio = Math.abs(infGain / splitInfo);
        return infGainRatio;
    }

    /**
     * measures the gini value of a contingency table
     * Finds row and column totals
     * Finds length and denominator
     * Performs gini algorithm
     * @param contingencyTable
     * @return gini
     */
    public static double measureGini(int[][] contingencyTable) {
        //TODO turn counters into a method
        List<Double> rowTotal = new ArrayList<>();
        List<Double> columnTotal = new ArrayList<>();
        double count;
        for (int i = 0; i < contingencyTable.length; i++) {
            count = 0;
            for (int j = 0; j < contingencyTable[i].length; j++) {
                count = count + contingencyTable[i][j];
            }
            rowTotal.add(count);
        }
        int[][] transposedContingencyTable;
        transposedContingencyTable = transpose(contingencyTable);
        for (int i = 0; i < transposedContingencyTable.length; i++) {
            count = 0;
            for (int j = 0; j < transposedContingencyTable[i].length; j++) {
                count = count + transposedContingencyTable[i][j];
            }
            columnTotal.add(count);
        }

        double len = 0;
        for (int i = 0; i < rowTotal.size(); i++) {
            len = len + rowTotal.get(i);
        }

        double denominator = 0;
        for (int i = 0; i < columnTotal.size(); i++) {
            denominator += columnTotal.get(i);
        }

        List<Double> impurity = new ArrayList<>();
        double tempImpurity = 0;
        for (int i = 0; i < columnTotal.size(); i++) {
            tempImpurity += Math.pow(columnTotal.get(i) / denominator, 2);
        }
        impurity.add(1 - tempImpurity);
        for (int i = 0; i < contingencyTable.length; i++) {
            tempImpurity = 0;
            for (int j = 0; j < contingencyTable[i].length; j++) {
                tempImpurity += Math.pow(contingencyTable[i][j] / rowTotal.get(i), 2);
            }
            impurity.add(1 - tempImpurity);
        }
        double gini = impurity.get(0);
        for (int i = 0; i < rowTotal.size(); i++) {
            gini -= (rowTotal.get(i) / denominator) * impurity.get(i+1);
        }
        return gini;
    }

    /**
     * measures the Chi squared value of a contingency table
     * Finds row and column totals
     * Finds length and denominator
     * calculates estimated values
     * performs Chi Squared algorithm
     * @param contingencyTable
     * @return Chi Squared
     */
    public static double measureChiSquared(int[][] contingencyTable) {
        List<Double> rowTotal = new ArrayList<>();
        List<Double> columnTotal = new ArrayList<>();
        double count;
        for (int i = 0; i < contingencyTable.length; i++) {
            count = 0;
            for (int j = 0; j < contingencyTable[i].length; j++) {
                count = count + contingencyTable[i][j];
            }
            rowTotal.add(count);
        }
        int[][] transposedContingencyTable;
        transposedContingencyTable = transpose(contingencyTable);
        for (int i = 0; i < transposedContingencyTable.length; i++) {
            count = 0;
            for (int j = 0; j < transposedContingencyTable[i].length; j++) {
                count = count + transposedContingencyTable[i][j];
            }
            columnTotal.add(count);
        }

        double len = 0;
        for (int i = 0; i < rowTotal.size(); i++) {
            len = len + rowTotal.get(i);
        }

        double denominator = 0;
        for (int i = 0; i < columnTotal.size(); i++) {
            denominator += columnTotal.get(i);
        }
        List<Double> Expected = new ArrayList<>();

        for (int i = 0; i < contingencyTable.length; i++) {
            for (int j = 0; j < contingencyTable[i].length; j++) {
                Expected.add(columnTotal.get(j) * ((rowTotal.get(i)) / denominator));
            }
        }
        double chiSquared = 0;
        int n = 0;
        for (int i = 0; i < contingencyTable.length; i++) {
            for (int j = 0; j < contingencyTable[i].length; j++) {
                chiSquared += Math.pow((contingencyTable[i][j] - Expected.get(n)),2) / Expected.get(n);
                n++;
            }
        }
        return chiSquared;
    }
}