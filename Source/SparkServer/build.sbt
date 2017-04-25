name := "SparkImages"

version := "1.0"

scalaVersion := "2.11.8"

scalacOptions ++= Seq(
  "-optimize",
  "-unchecked",
  "-deprecation"
)

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0" % "provided",
  "org.apache.spark" %% "spark-streaming" % "1.6.0",
  "org.apache.spark" %% "spark-mllib" % "1.6.0",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test",
  "org.bytedeco" % "javacpp" % "1.3.1",
  "org.bytedeco" % "javacv" % "1.3.1",
  "org.bytedeco" % "javacpp-presets" % "1.3",
  "org.bytedeco" % "javacpp-presets-platform" % "1.3",
  "net.databinder" %% "unfiltered-filter" % "0.8.3",
  "net.databinder" %% "unfiltered-jetty" % "0.8.3",
  "net.databinder" %% "unfiltered-directives" % "0.8.3",
  "com.levigo.jbig2" % "levigo-jbig2-imageio" % "1.6.5"
)

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "scala-tools" at "https://oss.sonatype.org/content/groups/scala-tools",
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
  "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
  "JavaCV maven repo" at "http://maven2.javacv.googlecode.com/git/",
  "JavaCPP maven repo" at "http://maven2.javacpp.googlecode.com/git/",
  "ImageIO" at "https://mvnrepository.com/artifact/com.levigo.jbig2/levigo-jbig2-imageio",
  "Maven Central" at "http://repo1.maven.org/maven2",
  Resolver.sonatypeRepo("public")
)
