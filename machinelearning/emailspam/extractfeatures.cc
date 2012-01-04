/* extract features from raw email (e.g., eml files) into an ARFF file
 * that can be used with Weka. adopt some techniques from research
 * papers as well as from Paul Graham's "a plan for spam" and "better
 * bayesian filtering" articles.
 *
 *
 * each raw email should be preprocessed such that the very first line
 * in the file is:
 *
 * Label: <L>
 *
 * where <L> is 1 if the email is spam and 0 if the email is ham.
 */

#ifdef USE_MIMETIC
#include <mimetic/mimetic.h>
#else
#include <vmime/vmime.hpp>
#include <vmime/platforms/posix/posixHandler.hpp>
#endif

#include <htmlcxx/html/ParserDom.h>
#include <htmlcxx/html/tree.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/assign/list_of.hpp> // for 'list_of()'
#include <boost/make_shared.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <assert.h>
#include <algorithm>
#include <getopt.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <cmath>
#include <iomanip>

static const char rcsid[] =
    "$Id$";

using namespace std;
using boost::regex;
using boost::regex_match;
using boost::regex_search;
using boost::lexical_cast;
using boost::trim;
using boost::split;
using boost::shared_ptr;

namespace fs = boost::filesystem;

#ifndef USE_MIMETIC
namespace vm = vmime;
#endif

// TRAIN_01242.eml in CSDMC2010_SPAM has over 3000 links
#define MAX_OCCURRENCES (3300)

#define ARRAY_LEN(arr) (sizeof (arr) / sizeof ((arr)[0]))

// these are some non-token features. for simplicity we consider them
// special "tokens". since we consider "*" and "&" separators, no
// normal token should have them, these special names should not
// conflict with any normal tokens
static const char g_hasHtmlFeature[] = "*&special_hasHtml";
static const char g_numLinksFeature[] = "*&special_numLinks";
static const char g_numImagesFeature[] = "*&special_numImages";
static const char g_numFontTagsFeature[] = "*&special_numFontTags";

static bool g_verbose = false;
static string g_tokenSeparators;

static string g_argv;

static string g_relationName = "email";

static set<string> g_ignoredHtmlTokens = boost::assign::list_of
  ("nbsp")
  ("quot")
  ("amp")
  ;

struct foo {
    bool operator() (long double i, long double j) { return (j < i); }
} descendingsorter;

template<typename T1, typename T2>
static inline bool
inMap(const std::map<T1, T2>& m, const T1& k)
{
    return m.end() != m.find(k);
}

template<typename T1>
static inline bool
inSet(const std::set<T1>& s, const T1& k)
{
    return s.end() != s.find(k);
}

static void
__attribute__ ((unused))
printTokens(const vector<string>& tokens)
{
    cout << " num elems: " << tokens.size() << endl;
    for (uint32_t i = 0; i < tokens.size(); ++i) {
        cout << "[" << tokens[i] << "] ";
    }
    cout << endl;
}

static int
countFiles(const char* dirpath)
{
    int file_count = 0;
    fs::path dp(dirpath);
    assert(fs::exists(dp));
    fs::directory_iterator it(dp);
    for (; it != fs::directory_iterator(); ++it) {
        if (fs::is_regular_file(*it)) {
            ++file_count;
        }
    }
    return file_count;
}

#if 0
template<typename T1>
static inline uint32_t
sumArray(const T1* valarray, const int count)
{
    int sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += valarray[i];
    }
    return sum;
}
#endif

/*
  each non-element in the array means an occurrence in a msg. return
  true if at least "threshold" elements are non-zero. "count" is
  number of elements.
 */
template<typename T1>
static inline bool
occursInAtLeast(const T1* valarray, const uint32_t count,
                const uint32_t threshold)
{
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (valarray[i] > 0) {
            ++sum;
            if (sum >= threshold) {
                return true;
            }
        }
    }
    return false;
}

static inline bool
isAllDigits(const string& s)
{
    BOOST_FOREACH(const char& c, s) {
        if (!isdigit(c)) {
            return false;
        }
    }
    return true;
}

#if USE_MIMETIC
static inline
bool is_known_mechanism(const string& mechanism)
{
    mimetic::istring m = mechanism;

    if(m == "base64" || m == "quoted-printable" || m == "binary" || 
       m == "7bit" || m == "8bit") {
        return true;
    }
    return false;
}
#endif

/* compute the mutual information between x and y.
 *
 * both arrays should have the same size: "N" number of elements
 */
template<typename T1>
static long double
computeMI(const T1* x, const T1* y, const int N)
{
    long double MI = 0.0;
    // the set of unique values
    set<T1> uniqx(x, x+N);
    set<T1> uniqy(y, y+N);

    BOOST_FOREACH(const T1& l1, uniqx) {
        BOOST_FOREACH(const T1& l2, uniqy) {
            // number of times ((x[i] == l1) && (y[i] == l2))
            int countBoth = 0;
            // number of times ((x[i] == l1))
            int countX = 0;
            // number of times ((y[i] == l2))
            int countY = 0;
            for (int i = 0; i < N; ++i) {
                if ((x[i] == l1) && (y[i] == l2)) {
                    ++countBoth;
                }
                if (x[i] == l1) {
                    ++countX;
                }
                if (y[i] == l2) {
                    ++countY;
                }
            }
            // P(X = l1, Y = l2)
            long double pboth = ((long double)countBoth) / ((long double)N);
            // P(X = l1)
            long double px = ((long double)countX) / ((long double)N);
            // P(Y = l2)
            long double py = ((long double)countY) / ((long double)N);
            if (pboth != 0 && px != 0 && py != 0) {
                MI += pboth * log2l(pboth / (px * py));
            }
        }
    }

    return MI;
}

void
parseMime(ifstream& file,
          const int& instNum,
          const int& totalNumInstances,
          const bool& useHtml,
          const bool& preserveCase,
          const set<string>* stopListWords,
          map<string, shared_ptr<uint16_t> >& tokenToCounts
    )
{
    if (g_verbose) {
        cout << " parsing mime of instNum " << instNum << endl;
    }

    vector<string> tokensInMsg;

#ifdef USE_MIMETIC
    mimetic::MimeEntity me(file);
#else
    vm::utility::inputStreamAdapter is(file);
    vm::string data;
    vm::utility::outputStreamStringAdapter os(data ) ;
    vm::utility::bufferedStreamCopy( is , os ) ;

    vm::ref<vm::message> msg = vm::create<vm::message>();
    msg->parse(data);

    static const vm::charset charset(vm::charsets::UTF_8);

    vm::messageParser mp(msg);
#endif

#ifdef USE_MIMETIC
    const mimetic::Header& _h = me.header(); // get header object
//	cout << _h.contentType() << endl;             // prints Content-Type
//    cout << "is multipath? " << h.contentType().isMultipart() << endl;
    const mimetic::MimeEntityList& _parts = me.body().parts(); // list of sub entities obj
	// cycle on sub entities list and print info of every item
	mimetic::MimeEntityList::const_iterator mbit = _parts.begin(), meit = _parts.end();
	for(; mbit != meit; ++mbit) {
//        const MimeEntity* me = *mbit;
        const mimetic::MimeEntity& me = **mbit;
        const mimetic::Header& mehdr = me.header();

        const string subType =
            boost::to_lower_copy(mehdr.contentType().subtype());

        if (!  ((subType == "plain") || (useHtml && subType == "html"))) {
            continue;
        }

        if (subType == "plain") {
            string text = me.body();

#else
    ////// extract tokens from subject
    string subj = mp.getSubject().getConvertedText(charset);
    //    cout << "msg subj: " << mp.getSubject().getConvertedText(charset) << endl;

    trim(subj);
    vector<string> subjTokens;
    split(subjTokens, subj, boost::is_any_of(g_tokenSeparators),
	  boost::token_compress_on);
    // have to manually massage/filter subj tokens before adding to
    // tokensInMsg
    BOOST_FOREACH(const string& _st_, subjTokens) {
        if (_st_.length() <= 0) {
            continue;
        }
        const string st = preserveCase ? _st_ : boost::to_lower_copy(_st_);
        if (stopListWords && inSet(*stopListWords, st)) {
            continue;
        }
        if (isAllDigits(st)) {
            continue;
        }
        string massagedSt = string("*&subj_") + st;
        tokensInMsg.push_back(massagedSt);
    }

//    assert (mp.getTextPartCount() < 10);

    // text part can only be either plain or html, nothing else.
    for (int i = 0; i < mp.getTextPartCount(); ++i) {
        vm::ref<const vm::textPart> tp = mp.getTextPartAt(i);

        if (tp->getType().getSubType() == vm::mediaTypes::TEXT_HTML &&
            !useHtml) {
            continue;
        }

        if (tp->getType().getSubType() == vm::mediaTypes::TEXT_PLAIN) {
            vm::ref<const vm::plainTextPart> ptp =
                tp.dynamicCast<const vm::plainTextPart>();
            const vm::ref<const vm::contentHandler> ch = ptp->getText();
            vm::string text;
            vm::utility::outputStreamStringAdapter ossa(text);
            ch->extract(ossa);

//            cout << "   plain text [[[" << text << "]]]" << endl;
#endif

            trim(text);
            vector<string> tokens;
            split(tokens, text, boost::is_any_of(g_tokenSeparators),
                  boost::token_compress_on);
            tokensInMsg.insert(tokensInMsg.end(), tokens.begin(), tokens.end());
//            printTokens(tokensInMsg);
        }
        else {
            assert (useHtml); // should not reach here is useHtml is
                              // false

#ifdef USE_MIMETIC
            assert (subType == "html");
            const mimetic::ContentTransferEncoding& cte =
                mehdr.contentTransferEncoding();
//            cout << cte.mechanism() << endl;
            assert (is_known_mechanism(cte.mechanism()));

            const mimetic::istring enc_algo = cte.mechanism();
            ostringstream oss;
            ostreambuf_iterator<char> oi(oss);

            if(enc_algo == mimetic::ContentTransferEncoding::base64) {
                mimetic::Base64::Decoder b64;
                decode(me.body().begin(), me.body().end(), b64 ,oi);
            } else if (enc_algo == mimetic::ContentTransferEncoding::quoted_printable) {
                mimetic::QP::Decoder qp;
                decode(me.body().begin(), me.body().end(), qp, oi);
            } else if (enc_algo == mimetic::ContentTransferEncoding::eightbit || 
                       enc_algo == mimetic::ContentTransferEncoding::sevenbit || 
                       enc_algo == mimetic::ContentTransferEncoding::binary) {
                copy(me.body().begin(), me.body().end(), oi);
            } else {
                cerr << "ERR: unknown encoding algorithm "
                     << enc_algo << endl;
            }

            const string htmlsrc = oss.str();
#else

            assert ((tp->getType().getSubType() == vm::mediaTypes::TEXT_HTML));
            tokenToCounts[g_hasHtmlFeature].get() [instNum] = 1;

            vm::ref<const vm::htmlTextPart> htp =
                tp.dynamicCast<const vm::htmlTextPart>();
            const vm::ref<const vm::contentHandler> ch = htp->getText();
            vm::string htmlsrc;
            vm::utility::outputStreamStringAdapter ossa(htmlsrc);
            ch->extract(ossa);

//            cout << "   the html src [[[" << htmlsrc << "]]]\n";

            //// !!! we want to look for links ("a" tags), images
            //// !!! ("img" tags), etc, but looks like these are not
            //// !!! vmime "objects", so we'll use Qt's html parser
            //// !!! instead
#endif

            htmlcxx::HTML::ParserDom parser;
            tree<htmlcxx::HTML::Node> dom = parser.parseTree(htmlsrc);

            tree<htmlcxx::HTML::Node>::iterator it = dom.begin();
            tree<htmlcxx::HTML::Node>::iterator end = dom.end();
            for (; it != end; ++it) {
                if (it->isTag()) {
                    const string tagName = boost::to_lower_copy(it->tagName());
                    if (tagName == "a") {
                        // need to see if it contains an href
                        it->parseAttributes();
                        map<string, string> attrs = it->attributes();
                        map<string, string>::const_iterator attrIt = attrs.begin();
                        // have to iterate and convert the attr names
                        // to lower case before comparing, otherwise
                        // might miss them.
                        for (; attrIt != attrs.end(); ++attrIt) {
                            if (boost::to_lower_copy(attrIt->first) == "href") {
                                tokenToCounts[g_numLinksFeature].get() [instNum] += 1;
                                assert (tokenToCounts[g_numLinksFeature].get() [instNum] < MAX_OCCURRENCES);
                                break;
                            }
                        }
                    }
                    else if (tagName == "img") {
                        tokenToCounts[g_numImagesFeature].get() [instNum] += 1;
                        assert (tokenToCounts[g_numImagesFeature].get() [instNum] < MAX_OCCURRENCES);
                    }
                    else if (tagName == "font") {
                        tokenToCounts[g_numFontTagsFeature].get() [instNum] += 1;
                        assert (tokenToCounts[g_numFontTagsFeature].get() [instNum] < MAX_OCCURRENCES);
                    }
                } /* end if isTag */
                else if (!it->isTag() && !it->isComment()) {
                    vector<string> tokens;
                    string text = it->text();
                    trim(text);
//                    cout << "  html text node: " << text << endl;
                    split(tokens, text,
                          boost::is_any_of(g_tokenSeparators),
                          boost::token_compress_on);
                    BOOST_FOREACH(const string& htmlToken, tokens) {
                        const string t = preserveCase ? htmlToken :
                                         boost::to_lower_copy(htmlToken);
                        if (inSet(g_ignoredHtmlTokens, t)) {
                            continue;
                        }
                        tokensInMsg.push_back(t);
                    }
                }
            }
        }
    }


    ///// all potential tokens in msg are now tokensInMsg,
    ///// filter/count them and update tokenToCounts
    BOOST_FOREACH(const string& _t_, tokensInMsg) {
        if (_t_.length() <= 0) {
//                    cout << " token with 0-length: [" << _t_ << "]" << endl;
            continue;
        }
        const string t = preserveCase ? _t_ : boost::to_lower_copy(_t_);
        if (stopListWords && inSet(*stopListWords, t)) {
            continue;
        }
        if (isAllDigits(t)) {
            continue;
        }
        // ok, usable token
        if (!inMap(tokenToCounts, t)) {
            // new token -> init
            shared_ptr<uint16_t> valarray(new uint16_t[totalNumInstances]);
            assert (valarray);
            bzero(valarray.get(), totalNumInstances * (sizeof (uint16_t)));
            tokenToCounts[t] = valarray;
        }
        tokenToCounts[t].get() [instNum] += 1;
#if 1
        assert (tokenToCounts[t].get() [instNum] < MAX_OCCURRENCES);
#else
        if (tokenToCounts[t].get() [instNum] >= MAX_OCCURRENCES) {
            cout << " WARN: in instNum " << instNum << ", token [" << t << "] counts: "
                 << tokenToCounts[t].get() [instNum] << endl;
        }
#endif
    }
}

void
extractTokens(const char* dirpath, const bool& rawEmail, const bool& useHtml,
              const bool& preserveCase,
              uint32_t topM,
              const char* stopListFile,
              const char* boolArffFile,
              const char* termFreqArffFile,
              const char* testCombinedDir,
              const char* testBoolArffFile,
              const char* testTermFreqArffFile)
{
    int totalNumInstances = countFiles(dirpath);
    uint16_t *instLabels = (uint16_t*)calloc(totalNumInstances, sizeof (instLabels[0]));
    bzero(instLabels, totalNumInstances * (sizeof instLabels[0]));
    map<string, shared_ptr<uint16_t> > tokenToCounts;

    // can't do both at the same time
    assert ((boolArffFile && !termFreqArffFile) || (!boolArffFile && termFreqArffFile));

    int instNum = 0;
    fs::path dp(dirpath);
    assert(fs::exists(dp));

    set<string> stopListWords;

    //// read in the stop list words
    if (stopListFile) {
        fs::path fp(stopListFile);
        if (!fs::exists(fp)) {
            cout << "stopListFile doesn't exist" << endl;
            exit(-1);
        }
        ifstream f(stopListFile);
        while (!f.eof()) {
            string line;
            std::getline(f, line);
            trim(line);
            if (line.length() == 0) {
                continue;
            }
            vector<string> tokens;
            split(tokens, line, boost::is_any_of(" ,"));
            BOOST_FOREACH(const string& t, tokens) {
                stopListWords.insert(t);
            }
        }
        cout << "number of stop list words: " << stopListWords.size() << endl;
    }

    /// some special "token"
    static const char* specialFeatures[] =
        {g_hasHtmlFeature,
         g_numLinksFeature,
         g_numImagesFeature,
         g_numFontTagsFeature,
        };
    {
        for (uint32_t i = 0; i < ARRAY_LEN(specialFeatures); ++i) {
            shared_ptr<uint16_t> valarray(new uint16_t[totalNumInstances]);
            assert (valarray);
            bzero(valarray.get(), totalNumInstances * (sizeof (uint16_t)));
            tokenToCounts[specialFeatures[i]] = valarray;
        }
    }


    fs::directory_iterator it(dp);

    for (; it != fs::directory_iterator(); ++it, ++instNum) {
        if (g_verbose) {
            cout << "file: " << it->string() << ", instNum: " << instNum << endl;
        }
        string line;
        ifstream f(it->string().c_str());

        //////////// the first line must be the label
        static const regex labelline_e("^Label: ([0-1]+)");
        boost::smatch labelline_m;

        std::getline(f, line);

        // get the label
        assert(regex_match(line, labelline_m, labelline_e));
        instLabels[instNum] = lexical_cast<int>(labelline_m[1]);

        if (rawEmail) {
            parseMime(f, instNum, totalNumInstances, useHtml, preserveCase,
                      stopListFile ? &stopListWords : NULL,
                      tokenToCounts
                      );
	    //            cout << "tokenToCounts.size() " << tokenToCounts.size() << endl;
        }
        else {
            // need to check code below before using non-raw parsing
            assert (false);
            /////////// the 2nd line must be the subject
            bool subjectline = true;

            while (!f.eof()) {
                std::getline(f, line);
                trim(line);
                if (line.length() == 0) {
                    continue;
                }

                vector<string> tokens;
                split(tokens, line, boost::is_any_of(" "));

                if (subjectline) {
                    subjectline = false;
                    assert (tokens[0] == "Subject:");
                    int oldsize = tokens.size();
                    tokens.erase(tokens.begin()); // remove "Subject:"
                    assert (tokens.size() == (oldsize - 1));
                    assert (tokens[0] != "Subject:");
                }

                BOOST_FOREACH(const string& t, tokens) {
                    assert (t.length() > 0);
                    // if the lower case is in the stop list, then we skip
                    if (stopListFile &&
                        /* preserveCase doesnt apply here */
                        inSet(stopListWords, boost::to_lower_copy(t)))
                    {
                        continue;
                    }
                    if (!inMap(tokenToCounts, t)) {
                        // new token -> init
                        // int *valarray = (int*)calloc(totalNumInstances,
                        //                              sizeof (valarray[0]));
                        shared_ptr<uint16_t> valarray(new uint16_t[totalNumInstances]);
                        assert (valarray);
                        bzero(valarray.get(), totalNumInstances * (sizeof (uint16_t)));
                        tokenToCounts[t] = valarray;
                    }
                    tokenToCounts[t].get() [instNum] += 1;
                    assert(tokenToCounts[t].get() [instNum] < MAX_OCCURRENCES);
                }

            }
        }
        f.close();
    }

#if 0
    {
        map<string, shared_ptr<uint16_t> >::const_iterator it = tokenToCounts.begin();
        for (; it != tokenToCounts.end(); ++it) {
            cout << " token: [" << it->first << "]" << endl;
        }
    }
#endif

    //// now tokenToCounts has all the tokens extracted from all
    //// messages, and each token is accompanied by an array of size
    //// totalNumInstances, with element ith being the number of
    //// occurrences of the token in the ith message

    {
        int oldsize = tokenToCounts.size();
        cout << "num tokens " << oldsize << endl;
        cout << "now we remove tokens that are too rare" << endl;
        int numremoved = 0;
        map<string, shared_ptr<uint16_t> >::iterator it = tokenToCounts.begin();
        while (it != tokenToCounts.end()) {
            uint16_t* valarray = it->second.get();
            if (!occursInAtLeast(valarray, totalNumInstances, 10)) {
                if (g_verbose) {
                    cout << " token " << it->first
                         << " too occurs in fewer than 10 emails -> removed\n";
                }
                ++numremoved;
                tokenToCounts.erase(it++); // must use post ++
            }
            else {
                ++it;
            }
        }
        assert (tokenToCounts.size() == (oldsize - numremoved));
        cout << "now " << tokenToCounts.size() << " tokens left" << endl;
    }

    ///
    /// now compute the mutual information for the tokens
    ///
    /// if we're doing bool, then change the term frequencies into
    /// bool "exists or not" (1 or 0)
    ///
    map<string, long double> tokenToMI;
    long double maxMI = -99999;
    vector<long double> MIs;
    string tokenWithMaxMI;
    {
        map<string, shared_ptr<uint16_t> >::iterator it = tokenToCounts.begin();
        for (; it != tokenToCounts.end(); ++it) {
            const string& t = it->first;
            uint16_t* valarray = it->second.get();

            if (boolArffFile) {
                // term freq -> bernoulli
                for (int i = 0; i < totalNumInstances; ++i) {
                    if (valarray[i] > 0) {
                        valarray[i] = 1;
                    }
                }
            }

            const long double MI = computeMI(
                valarray, instLabels, totalNumInstances);
            tokenToMI[t] = MI;
            if (MI > maxMI) {
                maxMI = MI;
                tokenWithMaxMI = t;
            }
            MIs.push_back(MI);
        }
    }

#if 0
    //////////////////////
    cout << tokenWithMaxMI << " " << tokenToMI[tokenWithMaxMI] << endl;
    for (int i = 0; i < totalNumInstances; ++i) {
        cout << tokenToCounts[tokenWithMaxMI] [ i ] << "  " << instLabels[i] << endl;
    }
    ////////////////////
    return;
#endif

    /// get the threshold MI
    sort(MIs.begin(), MIs.end(), descendingsorter);
    if (topM >= MIs.size()) {
        topM = MIs.size();
    }
    long double thresholdMI = MIs[topM - 1];

    /////
    cout << "only keep tokens with at least thresholdMI...\n";
    {
        int oldsize = tokenToCounts.size();
        cout << "num tokens " << oldsize << endl;
        int numremoved = 0;
        map<string, shared_ptr<uint16_t> >::iterator it = tokenToCounts.begin();
        while (it != tokenToCounts.end()) {
            const string& t = it->first;
            assert (inMap(tokenToMI, t));
            if (tokenToMI[t] < thresholdMI) {
                ++numremoved;
                tokenToCounts.erase(it++); // must use post ++
            }
            else {
                ++it;
            }
        }
        assert (tokenToCounts.size() == (oldsize - numremoved));
        cout << "now " << tokenToCounts.size() << " tokens left" << endl;
    }

    // sorted list of tokens we're interested in
    vector<string> finalTokenList;
    {
        map<string, shared_ptr<uint16_t> >::const_iterator it = tokenToCounts.begin();
        for (; it != tokenToCounts.end(); ++it) {
            finalTokenList.push_back(it->first);
        }
    }
    sort(finalTokenList.begin(), finalTokenList.end());

    /*##########################################
     *
     * create the arff files
     *
     ##########################################
    */

    map<string, ofstream*> arffFiles;
    if (boolArffFile) {
        assert (!termFreqArffFile);
        arffFiles["bool"] = new ofstream(boolArffFile);
    }
    else {
        assert (termFreqArffFile);
        arffFiles["termFreq"] = new ofstream(termFreqArffFile);
    }

    for (map<string, ofstream*>::iterator it = arffFiles.begin();
         it != arffFiles.end(); ++it)
    {
        assert (it->second != NULL);
        ofstream& arfffile = *(it->second);

//        const char* attrType = it->first == "bool" ? "{0,1}" : "NUMERIC";
        const bool boolattr = it->first == "bool";

        arfffile << "@RELATION \"" << g_relationName << "\"\n\n";

        // write comments
        arfffile << "% Revision = " << rcsid << endl;
        arfffile << "%" << endl;
        arfffile << "% argv: " << g_argv << endl;
        arfffile << "%" << endl;
        arfffile << "% finished: " << boost::posix_time::second_clock::local_time() << endl;
        arfffile << "%" << endl;
        arfffile << "% mode = " << it->first << endl;
        arfffile << "% dirpath = " << dirpath << endl;
        arfffile << "% stopListFile = " << (stopListFile ? stopListFile : "<none>") << (stopListFile ? (" (" + lexical_cast<string>(stopListWords.size()) + string(" words)")) : string()) << endl;
        arfffile << "% topM = " << topM << endl;
        arfffile << "% rawEmail = " << rawEmail << endl;
        arfffile << "% useHtml = " << useHtml << endl;
        arfffile << "% preserveCase = " << preserveCase << endl;
        arfffile << "%" << endl;
        arfffile << "% thresholdMI = " << setiosflags(ios::fixed) << setprecision(4) << thresholdMI << endl;
        arfffile << "% maxMI = " << setiosflags(ios::fixed) << setprecision(4) << maxMI << endl;
        arfffile << endl;

        // to avoid any issue with weka attribute names, we give each token
        // the attribute name "token_<num>" where <num> the token's index
        // in the finalTokenList + 1

        // first write the mapping (in comments)
        for (int num = 1; num < (finalTokenList.size() + 1); ++num) {
            arfffile << "% token_" << setfill('0') << setw(5) << num
                     << " -> [" << finalTokenList[num-1]
                     << "] (MI = " << setiosflags(ios::fixed) << setprecision(4) << tokenToMI[finalTokenList[num-1]]
                     << ")\n";
        }
        arfffile << endl;

        for (int num = 1; num < (finalTokenList.size() + 1); ++num) {
            arfffile << "@ATTRIBUTE  token_" << setfill('0') << setw(5) << num
                     << "  " << (boolattr ? "{0,1}" : "NUMERIC") << endl;
        }

        arfffile << "@ATTRIBUTE  class  {0,1}\n";

        // ################### write the data section

        arfffile << "\n@DATA\n";

        for (int instNum = 0; instNum < totalNumInstances; ++instNum) {
            BOOST_FOREACH(const string& t, finalTokenList) {
                int val = 0;
                if (boolattr) {
                    if (tokenToCounts[t].get()[instNum] > 0) {
                        // if bool attr, then any positive is 1
                        val = 1;
                    }
                }
                else {
                    // non-bool attr, use the term freq
                    val = tokenToCounts[t].get() [instNum];
                }

                arfffile << val << ",";
            }

            // output the instance label
            arfffile << instLabels[instNum] << endl;
        }

        arfffile.close();
    }


    /***********
     * now if there is testCombinedDir, then extract the now chosen
     * tokens from it.
     */
    if (testCombinedDir) {
        // can't do both at the same time
        assert ((testBoolArffFile && !testTermFreqArffFile) ||
                (!testBoolArffFile && testTermFreqArffFile));

        totalNumInstances = countFiles(testCombinedDir);
        free(instLabels);
        instLabels = (uint16_t*)
                     calloc(totalNumInstances, sizeof (instLabels[0]));
        assert(instLabels);
        bzero(instLabels, totalNumInstances * (sizeof instLabels[0]));
        {
            /// reinit the counts in tokenToCounts
            map<string, shared_ptr<uint16_t> >::iterator it =
                tokenToCounts.begin();
            for (; it != tokenToCounts.end(); ++it) {
                const string& t = it->first;
                shared_ptr<uint16_t> valarray(new uint16_t[totalNumInstances]);
                assert (valarray);
                bzero(valarray.get(), totalNumInstances * (sizeof (uint16_t)));
                // does this leak the old value? dont think so, but dont care.
                tokenToCounts[t].reset();
            tokenToCounts[t] = valarray;
            }
        }

        instNum = 0;
        fs::path testdp(testCombinedDir);
        assert(fs::exists(testdp));
        fs::directory_iterator testdp_it(testdp);
        for (; testdp_it != fs::directory_iterator(); ++testdp_it, ++instNum) {
            if (g_verbose) {
                cout << "file: " << testdp_it->string() << ", instNum: " << instNum << endl;
            }
            string line;
            ifstream f(testdp_it->string().c_str());

            //////////// the first line must be the label
            static const regex labelline_e("^Label: ([0-1]+)");
            boost::smatch labelline_m;

            std::getline(f, line);

            // get the label
            assert(regex_match(line, labelline_m, labelline_e));
            instLabels[instNum] = lexical_cast<int>(labelline_m[1]);

            if (rawEmail) {
                parseMime(f, instNum, totalNumInstances, useHtml, preserveCase,
                          stopListFile ? &stopListWords : NULL,
                          tokenToCounts
                    );
            }
            else {
                assert (false);
            }

            f.close();
        }

        //// create the arrf files
        if (testBoolArffFile) {
            assert (!testTermFreqArffFile);
            delete arffFiles["bool"];
            arffFiles["bool"] = new ofstream(testBoolArffFile);
        }
        else {
            assert (testTermFreqArffFile);
            delete arffFiles["termFreq"];
            arffFiles["termFreq"] = new ofstream(testTermFreqArffFile);
        }

        for (map<string, ofstream*>::iterator it = arffFiles.begin();
             it != arffFiles.end(); ++it)
        {
            assert (it->second != NULL);
            ofstream& arfffile = *(it->second);

//        const char* attrType = it->first == "bool" ? "{0,1}" : "NUMERIC";
            const bool boolattr = it->first == "bool";

            arfffile << "@RELATION \"TEST_" << g_relationName << "\"\n\n";

            // write comments
            arfffile << "% !!! this is for TESTING !!!" << endl;
            arfffile << "%" << endl;
            arfffile << "% Revision = " << rcsid << endl;
            arfffile << "%" << endl;
            arfffile << "% argv: " << g_argv << endl;
            arfffile << "%" << endl;
            arfffile << "% finished: " << boost::posix_time::second_clock::local_time() << endl;
            arfffile << "%" << endl;
            arfffile << "% mode = " << it->first << endl;
            arfffile << "% testCombinedDir = " << testCombinedDir << endl;
            arfffile << "% stopListFile = " << (stopListFile ? stopListFile : "<none>") << (stopListFile ? (" (" + lexical_cast<string>(stopListWords.size()) + string(" words)")) : string()) << endl;
            arfffile << "% topM = " << topM << endl;
            arfffile << "% rawEmail = " << rawEmail << endl;
            arfffile << "% useHtml = " << useHtml << endl;
            arfffile << "% preserveCase = " << preserveCase << endl;
            arfffile << "%" << endl;
            arfffile << "% thresholdMI (only from training data) = " << setiosflags(ios::fixed) << setprecision(4) << thresholdMI << endl;
            arfffile << "% maxMI (only from training data) = " << setiosflags(ios::fixed) << setprecision(4) << maxMI << endl;
            arfffile << endl;

            // to avoid any issue with weka attribute names, we give each token
            // the attribute name "token_<num>" where <num> the token's index
            // in the finalTokenList + 1

            // first write the mapping (in comments)
            for (int num = 1; num < (finalTokenList.size() + 1); ++num) {
                arfffile << "% token_" << setfill('0') << setw(5) << num
                         << " -> [" << finalTokenList[num-1]
                         << "] (MI = " << setiosflags(ios::fixed) << setprecision(4) << tokenToMI[finalTokenList[num-1]]
                         << ")\n";
            }
            arfffile << endl;

            for (int num = 1; num < (finalTokenList.size() + 1); ++num) {
                arfffile << "@ATTRIBUTE  token_" << setfill('0') << setw(5) << num
                         << "  " << (boolattr ? "{0,1}" : "NUMERIC") << endl;
            }

            arfffile << "@ATTRIBUTE  class  {0,1}\n";

            // ################### write the data section

            arfffile << "\n@DATA\n";

            for (int instNum = 0; instNum < totalNumInstances; ++instNum) {
                BOOST_FOREACH(const string& t, finalTokenList) {
                    int val = 0;
                    if (boolattr) {
                        if (tokenToCounts[t].get()[instNum] > 0) {
                            // if bool attr, then any positive is 1
                            val = 1;
                        }
                    }
                    else {
                        // non-bool attr, use the term freq
                        val = tokenToCounts[t].get() [instNum];
                    }

                    arfffile << val << ",";
                }

                // output the instance label
                arfffile << instLabels[instNum] << endl;
            }

            arfffile.close();
        }
    }
        
    return;
}

/* -------------------------------------------------------- */

void
initTokenSeparators()
{
    /*
      from paul graham's "a plan for spam":

      I currently consider alphanumeric characters, dashes,
      apostrophes, and dollar signs to be part of tokens, and
      everything else to be a token separator. (There is probably room
      for improvement here.) I ignore tokens that are all digits, and
      I also ignore html comments, not even considering them as token
      separators.

    */
    for (int c = 1; c < 256; ++c) {
        if (!isalpha(c) && !isdigit(c) && c != '-' && c != '\'' && c != '$') {
            g_tokenSeparators += c;
        }
    }
//    cout << " token separators:\n" << g_tokenSeparators << endl << endl;
}

/* -------------------------------------------------------- */

int main(int argc, char *argv[])
{
#ifndef USE_MIMETIC
    vm::platform::setHandler <vm::platforms::posix::posixHandler>();
#endif

    initTokenSeparators();

    assert(false == isAllDigits("1234."));
    assert(false == isAllDigits("12a34"));
    assert(false == isAllDigits("12X34"));
    assert(false == isAllDigits("1-234"));
    assert(false == isAllDigits("123$4"));
    assert(true == isAllDigits("1"));
    assert(true == isAllDigits("0"));
    assert(true == isAllDigits("0123456789"));

    for (int i = 0; i < argc; ++i) {
        g_argv += string(argv[i]) + " ";
    }

    const char* combinedDir = NULL;
    const char* boolArffFile = NULL;
    const char* termFreqArffFile = NULL;
    const char* stopListFile = NULL;
    const char* testCombinedDir = NULL;
    const char* testBoolArffFile = NULL;
    const char* testTermFreqArffFile = NULL;
    uint32_t topM = 0;
    bool rawEmail = false;
    bool useHtml = false;
    bool preserveCase = false;

    int opt;
    int long_index;

    struct option long_options[] = {
        {"combinedDir", required_argument, 0, 1001},
        {"topM", required_argument, 0, 1003},
        {"stopListFile", required_argument, 0, 1004},
        {"boolArffFile", required_argument, 0, 1005},
        {"termFreqArffFile", required_argument, 0, 1006},
        {"rawEmail", no_argument, 0, 1007},
        {"useHtml", no_argument, 0, 1008},
        {"verbose", no_argument, 0, 1009},
        {"relationName", required_argument, 0, 1010},
        {"preserveCase", no_argument, 0, 1011},

        /* the files in "testCombinedDir" are not used in selecting
         * which features to use, but otherwise should be in same
         * format as those in "combinedDir". once the features are
         * chosen, will extract those features from testCombinedDir
         * and write to either "testBoolArffFile" or
         * "testTermFreqArffFile"
         */
        {"test-combinedDir", required_argument, 0, 1012},
        {"test-boolArffFile", required_argument, 0, 1013},
        {"test-termFreqArffFile", required_argument, 0, 1014},
        {0, 0, 0, 0},
    };

    while ((opt = getopt_long(argc, argv, "", long_options, &long_index)) != -1)
    {
        switch (opt) {
        case 0:
            if (long_options[long_index].flag != 0) {
                break;
            }
            cout << "option " << long_options[long_index].name;
            if (optarg) {
                cout << " with arg " << optarg;
            }
            cout << endl;
            break;

        case 1001:
            combinedDir = optarg;
            break;

        case 1003:
            topM = strtol(optarg, NULL, 10);
            break;

        case 1004:
            stopListFile = optarg;
            break;

        case 1005:
            boolArffFile = optarg;
            break;

        case 1006:
            termFreqArffFile = optarg;
            break;

        case 1007:
            rawEmail = true;
            break;

        case 1008:
            useHtml = true;
            break;

        case 1009:
            g_verbose = true;
            break;

        case 1010:
            g_relationName = optarg;
            break;

        case 1011:
            preserveCase = true;
            break;

        case 1012:
            testCombinedDir = optarg;
            break;

        case 1013:
            testBoolArffFile = optarg;
            break;

        case 1014:
            testTermFreqArffFile = optarg;
            break;

        default:
            exit(-1);
            break;
        }
    }

    if (!rawEmail) {
        cout << "need to check code again if wanna use non-raw email files\n";
        exit(-1);
    }

    assert (combinedDir != NULL);
    assert ((boolArffFile != NULL) || (termFreqArffFile != NULL));
    assert (topM > 0);

    if (testCombinedDir) {
        assert (testBoolArffFile || testTermFreqArffFile);
        if (testBoolArffFile) {
            assert (!testTermFreqArffFile);
        }
    }

    extractTokens(combinedDir, rawEmail, useHtml, preserveCase,
                  topM, stopListFile,
                  boolArffFile, termFreqArffFile,
                  testCombinedDir, testBoolArffFile, testTermFreqArffFile);
    return 0;
}
