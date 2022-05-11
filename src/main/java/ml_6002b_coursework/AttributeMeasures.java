package ml_6002b_coursework;
import scala.tools.nsc.JarRunner;

import java.lang.Math;
import java.util.ArrayList;
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
        /** Table representing 0 as False and 1 as True
         * Islay is represented as 0 and Speyside is represented as 1
         * This will only work if 2 whiskey regions are used
         */

        //
        int[][] contingenyTable = new int[][]{
                {4, 0,},
                {1, 5,},
        };

        /*int[][] whiskeyData = new int[][]{
                {1, 0, 1, 0},
                {1, 1, 1, 0},
                {1, 0, 0, 0},
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 1, 1, 1},
                {0, 1, 1, 1},
                {0, 1, 1, 1},
                {0, 0, 1, 1},
                {0, 0, 1, 1},
        };
        */
        System.out.println("measure Information Gain for Peaty = "+measureInformationGain(contingenyTable));
        System.out.println("measure Information Gain Ratio for Peaty = "+measureInformationGainRatio(contingenyTable));
        System.out.println("measure Gini for Peaty = "+measureGini(contingenyTable));
        System.out.println("measure Chi Squared  for Peaty = "+measureChiSquared(contingenyTable));


    }

    public static double measureInformationGain(int[][] contingencyTable) {
        List<Double> rowTotal = new ArrayList<>();
        List<Double> columnTotal = new ArrayList<>();
        System.out.println("contTabLen"+contingencyTable.length);

        double count;
        for (int i = 0; i < contingencyTable.length; i++) {
            count = 0;
            System.out.println("bruh1");
            for (int j = 0; j < contingencyTable[i].length; j++) {

                count = count + contingencyTable[i][j];
            }
            rowTotal.add(count);
        }
        contingencyTable = transpose(contingencyTable);
        for (int i = 0; i < transpose(contingencyTable).length; i++) {
            count = 0;
            System.out.println(contingencyTable.length+" column");
            for (int j = 0; j < contingencyTable[i].length; j++) {
                count = count + contingencyTable[i][j];
            }
            columnTotal.add(count);
        }
        contingencyTable = transpose(contingencyTable);
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

        /**Entropy calculation proivided ML Lecture 2, page 25 and 26
         * -(5/10 log (5/10) + 5/10 log(5/10)) = 1
         *
         *Code only works if 2 possible whiskey types - if more are provided will need to adjust "1-islayChance"
         **/

        // chance iterates through all the potential outcomes and turns them into fractions based on the total
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
        // in test case: -(0.5log(0.5) + 0.5log(0.5) = 1

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
        //calculates the total datapoints provided
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
        contingencyTable = transpose(contingencyTable);
        for (int i = 0; i < transpose(contingencyTable).length; i++) {
            count = 0;
            for (int j = 0; j < contingencyTable[i].length; j++) {
                count = count + contingencyTable[i][j];
            }
            columnTotal.add(count);
        }
        contingencyTable = transpose(contingencyTable);

        double len = 0;
        for (int i = 0; i < rowTotal.size(); i++) {
            len = len + rowTotal.get(i);
        }

        double denominator = 0;
        for (int i = 0; i < columnTotal.size(); i++) {
            denominator += columnTotal.get(i);
        }

        List<Double> P = new ArrayList<>();
        double tempP = 0;
        for (int i = 0; i < columnTotal.size(); i++) {
            tempP += Math.pow(columnTotal.get(i) / denominator, 2);
        }
        P.add(1 - tempP);
        for (int j = 0; j < columnTotal.size(); j++) {
            tempP = 0;

            for (int i = 0; i < rowTotal.size(); i++) {
                tempP += Math.pow(contingencyTable[j][i] / rowTotal.get(j), 2);
            }
            P.add(1 - tempP);
        }
        double gini = P.get(0);
        for (int i = 0; i < rowTotal.size(); i++) {
            gini -= (rowTotal.get(i) / denominator) * P.get(i+1);
        }
        return gini;
    }

    public static double measureChiSquared(int[][] contingencyTable) {
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
        contingencyTable = transpose(contingencyTable);
        for (int i = 0; i < transpose(contingencyTable).length; i++) {
            count = 0;
            for (int j = 0; j < contingencyTable[i].length; j++) {
                count = count + contingencyTable[i][j];
            }
            columnTotal.add(count);
        }
        contingencyTable = transpose(contingencyTable);

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