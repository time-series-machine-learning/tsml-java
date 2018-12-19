/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.TreeSet;

/**
 *
 * @author ajb
 */
public class FilePermissionsTest {
    static String[] names = {"Tony,Michael,Paul,Gavin,Jason"};
    
    public static void main(String[] args) throws IOException {
//Test reading and writing from MOSNET
       String read="//gpfs/home/ajb/Results/TunedSVMPolynomial/Predictions/";
       String write="//gpfs/home/ajb/Testy/"+args[0];
       File f=new File(write);
       if(!f.isDirectory())
           f.mkdirs();
//Read through a  results set for fold0, adding up the train accuracies
        double accSum=0;
        int count=0;
        for(String str:DataSets.UCIContinuousFileNames){
//           
            if(new File(read+str).isDirectory()){
                File f2=new File(read+str+"/testFold0.csv");
                if(f2.exists()){
                    count++;
                    InFile inf=new InFile(read+str+"/testFold0.csv");
                    String t=inf.readLine();
                    t=inf.readLine();
                    accSum+=inf.readDouble();
                }
            }
        }
//Write to your own directory
        OutFile out = new OutFile(write+"/"+"MosnetTest.csv");
        out.writeLine("Count="+count+"  acc sum ="+accSum);
//Try reading others. Get directory names
        for(String str:names){
                if(!names.equals(args[0])){
                f=new File("//gpfs/home/ajb/Testy/"+str+"/"+"FileUseTest.csv");
                if(f.exists()){
                    InFile tt=new InFile("//gpfs/home/ajb/Testy/"+str+"/"+"MosnetTest.csv");
                    out.writeLine(names+":"+tt.readLine());
                }
            }
        }
        out.closeFile();
       f=new File(write+"/"+"MosnetTest.csv");
       TreeSet<PosixFilePermission> perms=new TreeSet<>();
       for(PosixFilePermission p:PosixFilePermission.values())
           perms.add(p);
       Path path=f.toPath();
       Files.setPosixFilePermissions(path, perms);
    }
}
