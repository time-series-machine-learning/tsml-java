package examples;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;


public class HARParser {

    static String outFileTrainArff = "C:\\Users\\fbu19zru\\code\\HAR_TRAIN.arff";
    static String outFileTrainTS = "C:\\Users\\fbu19zru\\code\\HAR_TRAIN.ts";
    static String outFileTestArff = "C:\\Users\\fbu19zru\\code\\HAR_TEST.arff";
    static String outFileTestTS = "C:\\Users\\fbu19zru\\code\\HAR_TEST.ts";

    static final int DIMENSIONS = 12;
    static final int MIN_FRAME_SIZE = 1000;
    static final int PADDING = 0;



    private static void saveData(FileWriter outFileWritterTS, FileWriter outFileWritterArff,
                                 double[][] instance, String classInstance) throws IOException {



        outFileWritterArff.write("'");
        for (int i=0;i<DIMENSIONS;i++){
            outFileWritterArff.write(Arrays.toString(instance[i]).replace("[", "")  //remove the right bracket
                    .replace("]", "").trim() );
            if (i<DIMENSIONS-1) outFileWritterArff.write("\\n");

            outFileWritterTS.write(Arrays.toString(instance[i])
                    .replace("[", "")  //remove the right bracket
                    .replace("]", "")
                    .trim() + ":");
        }

        outFileWritterArff.write("',"+classInstance+"\n");
        outFileWritterTS.write(classInstance + System.lineSeparator());

    }

    private static void saveDataInstance( FileWriter outFileWritterTS, FileWriter outFileWritterArff,
                                          ArrayList<double[]> instances, String label) throws IOException {

        double[][] instance = new double[DIMENSIONS][MIN_FRAME_SIZE];
        int i=0;

        for (int index = 0 ;index<instances.size();index++){
            double[] frame = instances.get(index);
            for (int j=0;j<DIMENSIONS;j++){
                instance[j][i] = frame[j];
            }
            i++;
            if (i==MIN_FRAME_SIZE){
                saveData(outFileWritterTS, outFileWritterArff,instance,label);
                instance = new double[DIMENSIONS][MIN_FRAME_SIZE];
                i=0;
                index -= PADDING;
            }
        }

    }




    private static void checkFile(String file, String label, FileWriter outFileWritterTS, FileWriter outFileWritterArff, String type){

  /*      int lineCount = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                lineCount++;

            }
            System.out.println(type + " " + lineCount);
        }catch (Exception e){
            e.printStackTrace();
        }
*/
        ArrayList<double[]> instances = new ArrayList<double[]>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;

            int j=0;
            while ((line = br.readLine()) != null) {
                if (j>0){
                    String[] spl = line.split(",");
                    double[] instance = new double[DIMENSIONS];
                    for (int i=0;i<DIMENSIONS;i++){
                        instance[i] = Double.parseDouble(spl[i+1]);
                    }
                    instances.add(instance);
                }
                j++;
            }
            saveDataInstance(outFileWritterTS, outFileWritterArff,instances,label);
        }catch (Exception e){
            e.printStackTrace();
        }

    }


    private static void writeHeaders(FileWriter fwTs, FileWriter fwArff) throws IOException {
        fwTs.write("@problemName HAR\n");
        fwTs.write("@timeStamps false\n");
        fwTs.write("@missing false\n");
        fwTs.write("@univariate false\n");
        fwTs.write("@dimensions 12\n");
        fwTs.write("@equalLength true\n");
        fwTs.write("@seriesLength "+MIN_FRAME_SIZE+"\n");
        fwTs.write("@classLabel true dws ups sit std wlk jog\n\n");
        fwTs.write("@data\n");

        fwArff.write("@relation 'HAR'\n");
        fwArff.write("@attribute relationalAtt relational\n");
        for (int i=0;i<DIMENSIONS;i++)
            fwArff.write(" @attribute att"+i+" numeric\n");
        fwArff.write("@end relationalAtt\n");
        fwArff.write("@attribute class {dws, ups, sit, std, wlk, jog}\n\n");
        fwArff.write("@data\n");

    }




    private static void executeFolder(String folder) throws IOException {
        File f = new File(folder);
        FileWriter outFileWritterTrainTS = new FileWriter(outFileTrainTS);
        FileWriter outFileWritterTrainARFF = new FileWriter(outFileTrainArff);
        FileWriter outFileWritterTestTS = new FileWriter(outFileTestTS);
        FileWriter outFileWritterTestARFF = new FileWriter(outFileTestArff);

        writeHeaders(outFileWritterTrainTS, outFileWritterTrainARFF);
        writeHeaders(outFileWritterTestTS, outFileWritterTestARFF);
        String[] folders = f.list();

        for (String fold : folders) {
            String label = fold.substring(0,3);
            int len = Integer.valueOf(fold.substring(4, fold.length()));
            File foldFiles = new File(folder + "\\" + fold);
            String[] files = foldFiles.list();
            for (String file: files){
                if (len<10)
                    checkFile(folder + "\\" + fold + "\\" + file, label, outFileWritterTrainTS, outFileWritterTrainARFF, "train");
                else
                    checkFile(folder + "\\" + fold + "\\" + file, label, outFileWritterTestTS, outFileWritterTestARFF, "test");
            }
        }
        outFileWritterTrainTS.close();
        outFileWritterTrainARFF.close();
        outFileWritterTestTS.close();
        outFileWritterTestARFF.close();
    }





    public static void main(String[] args){
        try {

            String folder = "C:\\Users\\fbu19zru\\code\\MotionSenseHAR\\A_DeviceMotion_data\\A_DeviceMotion_data";

            executeFolder(folder);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}