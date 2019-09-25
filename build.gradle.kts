plugins {
    java
    application
}

repositories {
    mavenCentral()    
    jcenter() 
}

dependencies {
    testCompile group: 'junit', name: 'junit', version: '4.12'
    testImplementation("junit:junit:4.12")
    implementation(fileTree("lib"))
}

application {
    mainClassName = "App"
}
