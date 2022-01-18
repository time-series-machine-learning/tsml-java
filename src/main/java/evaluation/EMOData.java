package evaluation;

import java.io.*;

public class EMOData {

    static String folder = "C:\\Users\\fbu19zru\\code\\MovementData" ;

    static String tsFileExercise = "C:\\Users\\fbu19zru\\code\\EMO_exercise_train.ts";
    static String arffFileExercise = "C:\\Users\\fbu19zru\\code\\EMO_exercise_train.arff";

    static String tsFilePain = "C:\\Users\\fbu19zru\\code\\EMO_pain_train.ts";
    static String arffFilePain = "C:\\Users\\fbu19zru\\code\\EMO_pain_train.arff";

    static int[] exerciseCount = new int[9];
    static int[] painCount = new int[3];

    private static void checkFile(String file, FileWriter outFileWritter){
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {

            String line;
            int count = 0;
            int lastExercise = -1;
            int lastPain = -1;

            while ((line = br.readLine()) != null) {
                String[] spl = line.split(",");
                if (lastExercise != Integer.parseInt(spl[30])){

                    if (count >= 180 && lastExercise != -1) {
                        exerciseCount[lastExercise] += count / 180;
                        painCount[lastPain] += count / 180;
                    }
                    count = 0;

                }else{
                    count++;
                }
                lastExercise = Integer.parseInt(spl[30]);
                lastPain = Integer.parseInt(spl[31]);



            }



        }catch (Exception e){
            e.printStackTrace();
        }
    }


    private static void executeFolder(FileWriter outFileWritter) throws IOException {
        File f = new File(folder);
        String[] files = f.list();

        for (String file : files) {
            checkFile(folder + "\\" + file, outFileWritter);
        }
        for (int i=0;i<9;i++){
            outFileWritter.write(i +", " + exerciseCount[i] + "\n");
        }

        for (int i=0;i<3;i++){
            outFileWritter.write(i +", " + painCount[i] + "\n");
        }
        outFileWritter.close();
    }

    private static void writeHeader(FileWriter outFileTSExercise, FileWriter outFileArffExercise,
                                    FileWriter outFileTSPain, FileWriter outFileArffPain) throws IOException {
        outFileTSExercise.write("@problemName EMOExercise\n");
        outFileTSExercise.write("@timeStamps false\n");
        outFileTSExercise.write("@missing false\n");
        outFileTSExercise.write("@univariate false\n");
        outFileTSExercise.write("@dimensions 30\n");
        outFileTSExercise.write("@equalLength true\n");
        outFileTSExercise.write("@seriesLength 180\n");
        outFileTSExercise.write("@classLabel true 0 1 2 3 4 5 6 7 8\n\n");
        outFileTSExercise.write("@data\n");

        outFileTSPain.write("@problemName EMOPain\n");
        outFileTSPain.write("@timeStamps false\n");
        outFileTSPain.write("@missing false\n");
        outFileTSPain.write("@univariate false\n");
        outFileTSPain.write("@dimensions 30\n");
        outFileTSPain.write("@equalLength true\n");
        outFileTSPain.write("@seriesLength 180\n");
        outFileTSPain.write("@classLabel true 0 1 2\n\n");
        outFileTSPain.write("@data\n");

        outFileArffExercise.write("@relation 'EMOExercise'\n");
        outFileArffExercise.write("@attribute relationalAtt relational\n");
        for (int i=0;i<180;i++)
            outFileArffExercise.write(" @attribute att"+i+" numeric\n");
        outFileArffExercise.write("@end relationalAtt\n");
        outFileArffExercise.write("@attribute class {0,1,2,3,4,5,6,7,8}\n\n");
        outFileArffExercise.write("@data\n");

        outFileArffPain.write("@relation 'EMOPain'\n");
        outFileArffPain.write("@attribute relationalAtt relational\n");
        for (int i=0;i<180;i++)
            outFileArffPain.write(" @attribute att"+i+" numeric\n");
        outFileArffPain.write("@end relationalAtt\n");
        outFileArffPain.write("@attribute class {0,1,2}\n\n");
        outFileArffPain.write("@data\n");

    }


    public static void main(String[] args){
        try {
            FileWriter outFileTSExercise = new FileWriter(tsFileExercise);
            FileWriter outFileArffExercise = new FileWriter(arffFileExercise);
            FileWriter outFileTSPain = new FileWriter(tsFilePain);
            FileWriter outFileArffPain = new FileWriter(arffFilePain);

            writeHeader(outFileTSExercise, outFileArffExercise, outFileTSPain, outFileArffPain);


           // writeBody(outFile);

            //executeFolder();


            outFileTSExercise.close();
            outFileArffExercise.close();
            outFileTSPain.close();
            outFileArffPain.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
