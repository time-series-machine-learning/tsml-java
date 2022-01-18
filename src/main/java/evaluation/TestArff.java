package evaluation;

import java.io.*;
import java.text.DecimalFormat;
import java.util.Arrays;

public class TestArff {
    static String file = "C:/Users/fbu19zru/code/test_day.csv";
    static String fileClass = "C:/Users/fbu19zru/code/class_test_day.csv";
    static String arffFile = "C:/Users/fbu19zru/code/test_day.arff";

    public static void main(String[] arg){


        try{
            FileWriter outFile = new FileWriter(arffFile);
            writeHeader(outFile);
            writeBody(outFile);
            outFile.close();
        }catch (Exception e){
            e.printStackTrace();
        }




    }
    private static void writeHeader(FileWriter fw) throws IOException {
        fw.write("@relation 'BasicMotions'\n");
        fw.write("@attribute relationalAtt relational\n");

        for (int i=0;i<200;i++)
            fw.write(" @attribute att"+i+" numeric\n");


        fw.write("@end relationalAtt\n");
        fw.write("@attribute class {1,2,3,4,5}\n\n");

        fw.write("@data\n");


    }
    private static void writeBody(FileWriter fw){
        DecimalFormat df = new DecimalFormat("#");
        df.setMaximumFractionDigits(20);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            BufferedReader brClass = new BufferedReader(new FileReader(fileClass));
            String line;
            String lineClass;
            while ((line = br.readLine()) != null) {
                lineClass = brClass.readLine();
                String[] spl = line.split(",");

                fw.write("'");
                for (int i=0;i<204;i++){
                    String[] subset = Arrays.copyOfRange(spl, i*200, (i+1)*200);
                    fw.write(Arrays.toString(subset).replace("[", "")  //remove the right bracket
                            .replace("]", "").trim() );
                    if (i<203) fw.write("\\n");
                }
                fw.write("',"+lineClass+"\n");




            }


        }catch (Exception e){
            e.printStackTrace();
        }
    }
}

