#!/usr/bin/perl

#################################################################################################################

# Copyright (C) 2021  Xiaowei Wang (email: xwang317@uic.edu), University of Illinois, Chicago
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
#################################################################################################################

use strict;
use warnings;
use FindBin 1.51 qw( $RealBin );
use lib $RealBin;
use Cwd;

###################################### SETTINGS ############################################################

my $version =         "v2.0";
my $file_dir =        "./";
my $result_dir =      "./result";
system("mkdir $result_dir") if !-e $result_dir;
my $temp_dir =        "./temp";
system("mkdir $temp_dir") if !-e $temp_dir;
my $classifier_dir =  "./ensemble";

my $inputFile;
my $result_file =     "./$temp_dir/gOligo_$version"."_prediction_result.xls";
my $oligo_file =      "./$temp_dir/oligos_$version".".txt";
my $predict_file =    "./$temp_dir/custom_prediction_result_$version".".txt";
my $outputFile =      "$result_dir/crisprdb_prediction_result.txt";

unlink $result_file if -e $result_file;
unlink $oligo_file if -e $oligo_file;
unlink $predict_file if -e $predict_file;
mkdir $result_dir if !(-e $result_dir);

my $minLength= 31;

################################### USER INPUTS ############################################################

my @inputs = @ARGV;
my $option = shift(@inputs);
&helpText if !@inputs;

my @sequences;
print "Welcome to CRISPRDB.\n\n";
if ($option eq '-e' or $option eq '--example'){
        my $sampleSelection = shift @inputs;
        
        die ("Please select a valid sample option.\n") unless $sampleSelection eq 'short' or $sampleSelection eq 'long' or $sampleSelection eq 'multiple';
        $inputFile = "./samples/test_sequence_short.fasta" if $sampleSelection eq 'short';
        $inputFile = "./samples/test_sequence_long.fasta" if $sampleSelection eq 'long';
        $inputFile = "./samples/test_sequence_multiple.fasta" if $sampleSelection eq 'multiple';
        print "Selected file: $inputFile\n";
        @sequences = importFasta($inputFile);        
}
elsif($option eq '-f' or $option eq '--file'){
        $inputFile = shift @inputs;
        print "Selected file: $inputFile\n";
        if (-e $inputFile && -r $inputFile && -f $inputFile && -T $inputFile){ #check to ensure file exists, readable, plain text
                open (INPUTCHECK,$inputFile);
                while (<INPUTCHECK>){
                        s/\s+$//;
                        die ("Please ensure that the file is in FASTA format.\n") if $_ !~ /^>/;
                        last;
                }
                close (INPUTCHECK);
                @sequences = importFasta($inputFile);                                                                            
        }else{
                print "Error: Please check to make sure the file \"$inputFile\" exists and is a readable plain text file.\n";
                exit;       
        }
}
elsif ($option eq '-s' or $option eq '--sequence'){
        my $submission = shift @inputs;
        ${$sequences[0]}{'seq'} = $submission;
        my $seqLength = length $submission;
        ${$sequences[0]}{'id'} = "submittedSequence|length_$seqLength";
}
else{
        &helpText;
}

################################### USER PREDICTION ########################################################

my $startTime = time();

# $submittedSeq =~ tr/ATCGU/atcgu/; $submittedSeq =~ tr/u/t/;
print "\n******************** Genome-Wide sgOligo Version $version Standard Output **************************\n";

open(OLI, ">$oligo_file");
print OLI "gRNA\t31mer\n";

my @feature;    # feature lines for SVM prediction
my @annotation; # annotation lines for mapping sgRNAs to gene annotations
my %oligo_pos; # to remove gene location redundancy from mulitple genomic loci
my $line_count = 0;
my $total_line_count = 0;
my $submittedSeq;
my $id;
my %id2Sequence;
foreach (@sequences){
        my $sequence = ${$_}{'seq'};
        $id = ${$_}{'id'}; 
        $submittedSeq = $sequence;
        $submittedSeq =~ tr/ATCGU/atcgu/; $submittedSeq =~ tr/u/t/;

        my $capital_seq = $sequence;
        $capital_seq =~ tr/atcgu/ATCGU/;
        $id2Sequence{$id}=$capital_seq;

        print "Error: Sequence contains bases other than A, T, C, G, or U. \n\tsgDesigner.2.0 will now proceed to the next sequence.\n\n" and next if $submittedSeq =~/[^atcg]/i;
        print "Error: Sequence is shorter than $minLength bases. \n\tsgDesigner.2.0 will now proceed to the next sequence.\n\n" and next if length ($submittedSeq)<$minLength;
        print "Error: Sequence is longer than 100,000 bases. \n\tsgDesigner.2.0 will now now proceed to the next sequence.\n\n" and next if length ($submittedSeq)>100000;

        my $submittedSeq_rc = dnaComplement($submittedSeq);

        identifyOligos($submittedSeq,"sense");
        identifyOligos($submittedSeq_rc,"antisense");
}
predict(\@annotation);

close OLI;
print "\n************* sgOligo selection process is done. Program completed successfully. *******************\n";

################################### USER OUTPUTS ########################################################

open(RESULT, $result_file) or die $!;
my %resultSeqs;
my %scoreList;
my $seqId;
while (<RESULT>) {
        s/\s+$//;
        next if $_ =~/^ensemble_prediction/;
        my @inline = split /\t/, $_;
        my $seq = substr($inline[6],5,20);
        my $seqSearch = $seq;
        my $orient = $inline[2];
        $seqId = $inline[1];
        $resultSeqs{$seq}{'orient'} = $orient;
        my $oligoSearch = $id2Sequence{$seqId};
        my $oligoLoc;
        $seqSearch = reverse($seq) and $seqSearch =~ tr/ATCG/TAGC/ if $orient eq 'antisense';
        my @pos1based;
        while ($oligoSearch =~ /$seqSearch/g) {
                $oligoLoc = pos($oligoSearch)-length($seqSearch)+1;
                push @pos1based,$oligoLoc;
        }

        my $location = join(", ",sort{$a<=>$b} @pos1based);
        $resultSeqs{$seq}{'location'} = $location;
        my $score= $inline[0];
        $scoreList{$seqId}{$seq} = $score;
}
close RESULT;

open(OUT, ">$outputFile") or die "$outputFile could not be opened for writing\n";

print OUT "seqId\tScore\tSequence\tOrientation\tPosition\n";
foreach my $sequenceId (sort keys %scoreList){

        foreach my $seq (sort {$scoreList{$sequenceId}{$b} <=> $scoreList{$sequenceId}{$a}} keys %{$scoreList{$sequenceId}}){                
                my $score = $scoreList{$sequenceId}{$seq};
                # next if $score<50;
                my $direction = $resultSeqs{$seq}{'orient'};
                my $position = $resultSeqs{$seq}{'location'};
                print OUT "$sequenceId\t$score\t$seq\t$direction\t$position\n";
        }
        
}
close OUT;

my $endTime = time();
my $finalTime = $endTime-$startTime;
print"\nResults have been printed in $outputFile. Program completed in $finalTime seconds.\n";

########################################################################################################################################################################
sub identifyOligos {

    my ($exon_plus,$orientation) = @_;
    
    my $dummyString = 'n'x6;
    $exon_plus = join("",$dummyString,$exon_plus,$dummyString);

    my $exon = substr($exon_plus, 6, length($exon_plus) - 12);

    my $seqStrand = $exon_plus;
    $seqStrand = dnaComplement($exon_plus) if $orientation eq 'antisense'; #return to the original sequence

    my @gg_pos_list;
    for (my $i =0; $i<length($exon);$i++){
        my $dinuc = substr($exon,$i,2);
        push @gg_pos_list, $i if $dinuc eq 'gg';
    }
    foreach my $gg_pos (@gg_pos_list) {
 
            my $oligo = substr($exon_plus, $gg_pos + 6 - 26, 31);
            next if $oligo =~ /n/g;
            
            my $matched_bases = $oligo;
            $matched_bases = dnaComplement($oligo) if $orientation eq 'antisense';
            
            my $cds_pos = index($seqStrand,$matched_bases)+1;

            ######## changed in V1.2 ############
            my $gRNA = substr($oligo, 5, 20);
            $gRNA =~ tr/atcgu/ATCGU/;
            $oligo =~ tr/atcgu/ATCGU/;
            my $output = "$gRNA\t$oligo";
            print OLI "$output\n";
            #####################################

            if ($output) {
                 $feature[$line_count] = $output;

                 my $exon_pos = $gg_pos;
                 $exon_pos = length($exon) - $gg_pos if $orientation eq "antisense";
                 $exon_pos += 1; # 1-based index position

                 $annotation[$line_count] = "$id\t$orientation\t$exon_pos\t$cds_pos\t".length($exon_plus)."\t$oligo";

                 $line_count++;
            }
            $total_line_count++;
    }
}

sub predict {

    my ($annotation_ref) = @_;
    my @annotation = @{$annotation_ref};
    my $line_count = 0;

    system("python3 $classifier_dir/ensemble.py");

    open(OUT, ">$result_file");
    print OUT "ensemble_prediction\tsequenceID\tOrientation\tPosition in Exon\tPosition in CDS\tCDS Length\tOligo Sequence\n";

    open (IN, "$predict_file");<IN>;
    while(<IN>){
      $_ =~ s/\s+$//;
      my $out = $_."\t".$annotation[$line_count];
      print OUT "$out\n";
      $line_count ++;
    }
    close IN;
    close OUT;
}

# return the self-complementary strand of the input sequence
sub dnaComplement {

    my ($sequence) = @_;
    $sequence =~ tr/atcgATCG/tagcTAGC/;
    $sequence = reverse($sequence);
    return $sequence;
}

sub importFasta {
    my ($fastaFile) = @_;
    my $tabFile = "$fastaFile $$.tab";
    fastaToTab($fastaFile, $tabFile);
    my @seq = importTabSeq($tabFile);
    unlink $tabFile if -e $tabFile;
    return @seq;
}

sub fastaToTab {
     my ($fastaFile, $tabFile) = @_;
     my $id = "";
     my $dna= "";
     my $lastLine = "";
     open (IN, "$fastaFile") || die("Can not open $fastaFile file for reading in fastaToTab sub!\n");
     open (OUT, ">$tabFile") || die("Can not open $tabFile file for writing!\n");

     while (<IN>) {
          s/\s+$//;
          next if ($_ !~ /\S/);
          if ($_ =~ /^\>/) {
               $id = $_;
               $id =~ s/^\>//;
               if ($lastLine =~ /^\>/) {   
                    $id .= $_;
               }
               else {
                    print OUT "\n" if ($dna ne ""); 
                    print OUT "$id\t";
                    $id = "";
               }
          }
          else {
               $_ =~ s/\s//g;
               $dna = $_;
               print OUT $dna;
          }
          $lastLine = $_;
     }
     close(IN);
     close(OUT);
}

sub importTabSeq {
     my ($tabFile) = @_;
     my @sequence = ();
     my $index = 0;
     open (IN, "$tabFile") || die("Cannot open $tabFile file for reading in importTab sub!\n");
     while (<IN>) {
          s/\s+$//;
          my ($id, $sequence) = split /\t/, $_;
          $sequence[$index]{'id'} = $id;
          $sequence =~ tr/A-Z/a-z/;
          $sequence[$index]{'seq'} = $sequence;
          $index++;
     }
     close(IN);
     return @sequence;
}

sub helpText{
        print "\n";
        
        print "USAGE:\n\tperl sgDesigner.pl [option] [path]\n\tperl sgDesigner.pl [option] [sequence]\n\n";
        print "SEQUENCE SUBMISSION:\n\t-s|--sequence <sequence>\n\t\t";
        print "Identifies sgRNA oligos from a single submitted sequence \n\t\tand provides a score for all potential active oligos. \n\t\tResults for submitted sequences will be printed to a \n\t\ttab-delimited text file.\n";
        print "\n\t\tExample: perl sgDesigner.pl -s acctgcgtggctcccctgagtggagt\n\n";
        
        print "FILE SUBMISSION:\n\t-f|--file <file>\n\t\t";
        print "Imports a FASTA file of sequences from <file> and \n\t\tidentifies potential sgRNA oligos for each submitted\n\t\tsequence. Resulting oligos are available in a tab-\n\t\tdelimited text file.\n";
        print "\n\t\tExample: perl sgDesigner.pl -f mySampleFile.fasta\n\n";
        
        print "EXAMPLE FILE SUBMISSION:\n\t-e|--example <short|long|multiple>\n\t\t";
        print "Uses the short, long, or multiple sample sequence in the \n\t\tsamples directory to generate sgRNA oligos.\n";
        print "\n\t\tExample: perl sgDesigner.pl -e short\n\n";
        
        print "HELP SCREEN:\n\t-h|--help \n\t\t";
        print "Brings up this help menu";
        print "\n\t\tExample: perl sgDesigner.pl -h\n\n";
        
        exit;
}