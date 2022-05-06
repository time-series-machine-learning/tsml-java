package ml_6002b_coursework;
import java.lang.Math;
/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */

    public static void main(String[] args) {
        /** Table representing 0 as False and 1 as True
         * Islay is represented as 0 and Speyside is represented as 1
         * This will only work if 2 whiskey regions are used
         */
        int[][] whiskeyData = new int[][]{
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
    }

    public static double measureInformationGain(int[][] whiskeyData) {
        double len = whiskeyData.length;
        double infGain = 0;
        int P1TrueCount = 0;
        int P1FalseCount = 0;
        int P2TrueCount = 0;
        int P2FalseCount = 0;
        int islayCount = 0;
        for (int i = 0; i < len; i++) {
            if (whiskeyData[i][0] == 1 && whiskeyData[i][3] == 1) {
                P1TrueCount++;
            }
            if (whiskeyData[i][0] == 1 && whiskeyData[i][3] == 0) {
                P1FalseCount++;
            }
            if (whiskeyData[i][0] == 0 && whiskeyData[i][3] == 1) {
                P2TrueCount++;
            }
            if (whiskeyData[i][0] == 0 && whiskeyData[i][3] == 0) {
                P2FalseCount++;
            }
            if (whiskeyData[i][3] == 1) {
                islayCount++;
            }
        }
        double speysideCount = len - islayCount;


        System.out.println(islayCount);

        double entropy;
        double Hx1 = 1.0;
        double Hx2 = 1.0;


        /**Entropy calculation proivided ML Lecture 2, page 25 and 26
         * -(5/10 log (5/10) + 5/10 log(5/10)) = 1
         *
         *Code only works if 2 possible whiskey types - if more are provided will need to adjust "1-islayChance"
         **/
        double islayChance = islayCount / len;
        double speysideChance = speysideCount / len;
        /** NOTE
         * All logorithms in this class are log base 2 :
         * The conversion is done using the Math.log giving natural log then dividing it by log 2
         */
        entropy = -(((islayChance) * ((Math.log(islayChance) / Math.log(2)))) +
                ( (speysideChance) * ( (Math.log(speysideChance) ) / Math.log(2))));

        double P1Len = P1TrueCount+P1FalseCount;
        double P2Len = P2TrueCount+P2FalseCount;

        //Hx1 = ...
        if(P1TrueCount != 0.0 || P1FalseCount != 0.0) {
            Hx1 = -(((P1TrueCount / P1Len) * ((Math.log(P1TrueCount / P1Len) / Math.log(2)))) +
                    ((P1FalseCount / P1Len) * ((Math.log(P1FalseCount / P1Len)) / Math.log(2))));
        }
        //Hx2 = ...
        if(P2TrueCount != 0.0 || P2FalseCount != 0.0) {
            Hx2 = -(((P2TrueCount / P2Len) * ((Math.log(P2TrueCount / P2Len) / Math.log(2)))) +
                    ((P2FalseCount / P2Len) * ((Math.log(P2FalseCount / P2Len)) / Math.log(2))));
        }
        infGain = entropy - (((P1Len/(P1Len+P2Len)*Hx1) -((P2Len/(P1Len+P2Len)*Hx2))));
        return infGain;
    }


 public double measureInformationGainRatio (int[][] whiskeyData){

    double infGain = measureInformationGain(whiskeyData);

    int PTrueCount = 0;
    int PFalseCount = 0;
    double splitInfo;
    int dataSize = whiskeyData.length;

    for (int i=0; i < dataSize; i++){
        if (whiskeyData[i][1] == 0) {
            PFalseCount++;
        }
        if (whiskeyData[i][1] == 1) {
            PTrueCount++;
        }
    }
    double PTotal = PTrueCount + PFalseCount;
    //Split Info calculation:
     splitInfo = (PTrueCount / (PTotal * (Math.log(PTrueCount / PTotal) / Math.log(2)))) +
                (PFalseCount / (PTotal * (Math.log(PFalseCount / PTotal) / Math.log(2))));

     double infGainRatio = infGain / splitInfo;
 return infGainRatio;
 }

 public static double measureGini (int[][] whiskeyData){
        //TODO turn counters into a method
     int P1TrueCount = 0;
     int P1FalseCount = 0;
     int P2TrueCount = 0;
     int P2FalseCount = 0;
     int islayCount = 0;
     double len = whiskeyData.length;
     for (int i = 0; i < len; i++) {
         if (whiskeyData[i][0] == 1 && whiskeyData[i][3] == 1) {
             P1TrueCount++;
         }
         if (whiskeyData[i][0] == 1 && whiskeyData[i][3] == 0) {
             P1FalseCount++;
         }
         if (whiskeyData[i][0] == 0 && whiskeyData[i][3] == 1) {
             P2TrueCount++;
         }
         if (whiskeyData[i][0] == 0 && whiskeyData[i][3] == 0) {
             P2FalseCount++;
         }
         if (whiskeyData[i][3] == 1) {
             islayCount++;
         }
     }

     double P1Count = P1TrueCount+P1FalseCount;
     double P2Count = P2TrueCount+P2FalseCount;
     double speysideCount = len - islayCount;
     double islayChance = islayCount / len;
     double speysideChance = speysideCount / len;

     double Px = 1 - Math.pow(P1Count, 2) + Math.pow(P2Count, 2);
     double P1 = 1 - Math.pow(P1TrueCount/P1Count,2) + Math.pow(P1FalseCount/P1Count,2);
     double P2 = 1 - Math.pow(P2TrueCount/P2Count,2) + Math.pow(P2FalseCount/P2Count,2);

     double gini = Px - islayChance/P1 - speysideChance/P2;
 return gini;
 }

 public static double measureChiSquared (int[][] whiskeyData){
     //TODO turn counters into a method
     int P1TrueCount = 0;
     int P1FalseCount = 0;
     int P2TrueCount = 0;
     int P2FalseCount = 0;
     int islayCount = 0;
     double len = whiskeyData.length;
     for (int i = 0; i < len; i++) {
         if (whiskeyData[i][0] == 1 && whiskeyData[i][3] == 1) {
             P1TrueCount++;
         }
         if (whiskeyData[i][0] == 1 && whiskeyData[i][3] == 0) {
             P1FalseCount++;
         }
         if (whiskeyData[i][0] == 0 && whiskeyData[i][3] == 1) {
             P2TrueCount++;
         }
         if (whiskeyData[i][0] == 0 && whiskeyData[i][3] == 0) {
             P2FalseCount++;
         }
         if (whiskeyData[i][3] == 1) {
             islayCount++;
         }
     }
     double speysideCount = len - islayCount;
     int PTrueCount = P1TrueCount+P2TrueCount;
     int PFalseCount = P1FalseCount+P2FalseCount;

     double P1TrueExpected = islayCount*(PTrueCount/len);
     double P1FalseExpected = islayCount*(PFalseCount/len);
     double P2TrueExpected = speysideCount*(PTrueCount/len);
     double P2FalseExpected = speysideCount*(PFalseCount/len);

     double chiSquared = (Math.pow((P1TrueCount - P1TrueExpected),2)) +
             Math.pow((P1FalseCount - P1FalseExpected),2) +
             Math.pow((P2TrueCount - P2TrueExpected),2 ) +
             Math.pow((P2FalseCount - P2FalseExpected),2 );
 return chiSquared;
 }

}


