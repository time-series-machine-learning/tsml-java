package examples;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;



public class Parser {

    static String FOLDER_DATA = "C:\\Users\\fbu19zru\\code\\MovementData\\ValidationCsv";
    static String FILE_DATA = "C:\\Users\\fbu19zru\\code\\MovementData\\trainCsv\\C56D.csv";
    static String outFileArff = "C:\\Users\\fbu19zru\\code\\EMO_TEST.arff";
    static String outFileTS = "C:\\Users\\fbu19zru\\code\\EMO_TEST.ts";
    static String CLASSES = "0 1 2";
    static String PROBLEM_NAME = "EmoPain";

    static final int DIMENSIONS = 30;
    static final int MIN_FRAME_SIZE = 200;
    static final int CLASS_INDEX = 31;
    static final int PADDING = 0;


    private static void saveData(FileWriter outFileWritterTS, FileWriter outFileWritterArff,
                                 double[][] instance, int classInstance) throws IOException {



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
                                          ArrayList<double[]> instances, int classInstance) throws IOException {

        double[][] instance = new double[DIMENSIONS][MIN_FRAME_SIZE];
        int i=0;

        for (int index = 0 ;index<instances.size();index++){
            double[] frame = instances.get(index);
            for (int j=0;j<DIMENSIONS;j++){
                instance[j][i] = frame[j];
            }
            i++;
            if (i==MIN_FRAME_SIZE){
                saveData(outFileWritterTS, outFileWritterArff,instance,classInstance);
                instance = new double[DIMENSIONS][MIN_FRAME_SIZE];
                i=0;
                index -= PADDING;
            }
        }

    }



    private static void checkFile(String file, FileWriter outFileWritterTS, FileWriter outFileWritterArff){

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            int count = 0;
            int lastClass = -1;
            ArrayList<double[]> instances = new ArrayList<double[]>();

            while ((line = br.readLine()) != null) {
                String[] spl = line.split(",");
                if (lastClass != Integer.parseInt(spl[CLASS_INDEX])){
                    if (count >= MIN_FRAME_SIZE && lastClass != -1) {
                        saveDataInstance(outFileWritterTS,outFileWritterArff, instances, lastClass);
                    }
                    count = 0;
                    instances.clear();
                }else{
                    double[] inst = new double[DIMENSIONS];
                    for (int i=0;i<DIMENSIONS;i++){
                        inst[i] = Double.valueOf(spl[i]);
                    }
                    instances.add(inst);
                    count++;
                }
                lastClass = Integer.parseInt(spl[CLASS_INDEX]);
            }
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    private static void checkFile(String file){

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            int count = 0;
            int lastClass = -1;


            while ((line = br.readLine()) != null) {
                String[] spl = line.split(",");
                if (lastClass!= -1 && lastClass != Integer.parseInt(spl[CLASS_INDEX])){
                    System.out.println(lastClass + " " + count);
                    count = 0;

                }else{
                    count++;
                }
                lastClass = Integer.parseInt(spl[CLASS_INDEX]);
            }
            System.out.println(lastClass + " " + count);
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    private static void writeHeaders(FileWriter fwTs, FileWriter fwArff) throws IOException {
        fwTs.write("@problemName "+PROBLEM_NAME+"\n");
        fwTs.write("@timeStamps false\n");
        fwTs.write("@missing false\n");
        fwTs.write("@univariate false\n");
        fwTs.write("@dimensions "+DIMENSIONS+"\n");
        fwTs.write("@equalLength true\n");
        fwTs.write("@seriesLength "+MIN_FRAME_SIZE+"\n");
        fwTs.write("@classLabel true "+CLASSES+"\n\n");
        fwTs.write("@data\n");

        fwArff.write("@relation '"+PROBLEM_NAME+"'\n");
        fwArff.write("@attribute relationalAtt relational\n");
        for (int i=0;i<DIMENSIONS;i++)
            fwArff.write(" @attribute att"+i+" numeric\n");
        fwArff.write("@end relationalAtt\n");
        fwArff.write("@attribute class {"+CLASSES+"}\n\n");
        fwArff.write("@data\n");

    }



    private static void checkFolder(String folder) throws IOException {
        File f = new File(folder);
        FileWriter outFileWriterTS = new FileWriter(outFileTS);
        FileWriter outFileWriterARFF = new FileWriter(outFileArff);

        writeHeaders(outFileWriterTS, outFileWriterARFF);
        String[] files = f.list();

        for (String file : files) {
            checkFile(folder + "\\" + file, outFileWriterTS, outFileWriterARFF);
        }

        outFileWriterTS.close();
        outFileWriterARFF.close();
    }

    private static void checkFolderTest(String folder) throws IOException {
        File f = new File(folder);
        String[] files = f.list();

        for (String file : files) {
            checkFile(folder + "\\" + file);
        }

    }





    public static void main(String[] args){
        try {



            checkFolder(FOLDER_DATA);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}