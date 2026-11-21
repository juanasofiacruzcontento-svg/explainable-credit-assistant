# Note sulla normativa UE per l'IA nel credito

Sintesi redatte per questo progetto, parafrasate e semplificate. Non sono il testo
ufficiale e non costituiscono consulenza legale. Testo ufficiale del Regolamento
(UE) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
Aggiornate a novembre 2026.

## [AI Act, Allegato III punto 5(b)] Il credit scoring è ad alto rischio

I sistemi di IA usati per valutare il merito creditizio delle persone fisiche o per
stabilire il loro punteggio di credito sono classificati come sistemi ad alto rischio.
Fanno eccezione i sistemi usati per individuare frodi finanziarie. La ragione è che
l'accesso al credito incide su risorse essenziali come casa ed elettricità e i
sistemi possono perpetuare discriminazioni storiche, ad esempio per origine, età,
disabilità o genere.

## [AI Act, Digital Omnibus] Quando si applicano gli obblighi

L'AI Act è entrato in vigore il 1 agosto 2024 con applicazione scaglionata. Gli
obblighi per i sistemi ad alto rischio dell'Allegato III, tra cui il credit scoring,
erano previsti dal 2 agosto 2026. Il pacchetto "Digital Omnibus on AI", approvato nel
2026, li ha rinviati al 2 dicembre 2027. L'obbligo di alfabetizzazione in materia di
IA del personale (art. 4) e i divieti dell'art. 5 si applicano già dal 2 febbraio 2025.

## [AI Act, art. 9] Sistema di gestione dei rischi

Il fornitore deve istituire un processo continuo per identificare, stimare e mitigare
i rischi prevedibili per salute, sicurezza e diritti fondamentali, lungo tutto il
ciclo di vita del sistema, e testare il sistema prima dell'immissione sul mercato.

## [AI Act, art. 10] Dati e governance dei dati

I dati di addestramento, validazione e test devono essere pertinenti, sufficientemente
rappresentativi e, per quanto possibile, privi di errori e completi. Vanno esaminati
per individuare possibili distorsioni (bias) che possano portare a discriminazioni e
vanno adottate misure per prevenirle e attenuarle. In casi strettamente necessari e con
garanzie, è consentito trattare categorie particolari di dati personali proprio per
rilevare e correggere i bias.

## [AI Act, art. 11 e 12] Documentazione tecnica e registrazione degli eventi

Prima dell'immissione sul mercato va redatta una documentazione tecnica che dimostri
la conformità (scopo, architettura, dati, metriche, limiti). Il sistema deve inoltre
registrare automaticamente gli eventi (log) durante il funzionamento, per garantire
la tracciabilità e il monitoraggio successivo.

## [AI Act, art. 13] Trasparenza verso chi usa il sistema

Il sistema deve essere progettato in modo che chi lo utilizza (il deployer, ad esempio
la banca) possa interpretarne l'output e usarlo correttamente. Le istruzioni per l'uso
devono indicare scopo, livello di accuratezza, limiti noti, rischi prevedibili e, se
pertinente, informazioni che aiutano a interpretare l'output.

## [AI Act, art. 14] Sorveglianza umana

I sistemi ad alto rischio devono poter essere supervisionati efficacemente da persone.
Chi supervisiona deve capirne capacità e limiti, essere consapevole del rischio di
affidarsi troppo all'output (automation bias), poter interpretare correttamente
l'output e poter decidere di non usarlo, ignorarlo o ribaltarlo.

## [AI Act, art. 15] Accuratezza, robustezza e cibersicurezza

Il sistema deve raggiungere un livello adeguato di accuratezza, robustezza e
cibersicurezza e mantenerlo per tutto il ciclo di vita. I livelli di accuratezza e le
metriche usate vanno dichiarati nelle istruzioni per l'uso. Per i sistemi che
continuano ad apprendere vanno ridotti i rischi di cicli di retroazione distorti.

## [AI Act, art. 26] Obblighi di chi utilizza il sistema (deployer)

La banca che usa un sistema ad alto rischio deve usarlo secondo le istruzioni,
affidare la sorveglianza umana a persone competenti, monitorarne il funzionamento,
conservare i log generati e informare le persone che sono soggette a decisioni prese
o supportate dal sistema.

## [AI Act, art. 27] Valutazione d'impatto sui diritti fondamentali

Prima di usare un sistema di credit scoring, i deployer interessati (tra cui chi
valuta il merito creditizio) devono effettuare una valutazione dell'impatto sui
diritti fondamentali: processi in cui il sistema è usato, categorie di persone
coinvolte, rischi specifici di danno, misure di sorveglianza umana e misure da
adottare se i rischi si concretizzano.

## [AI Act, art. 86] Diritto alla spiegazione delle decisioni individuali

Chi è soggetto a una decisione presa dal deployer sulla base dell'output di un sistema
ad alto rischio dell'Allegato III, che produce effetti giuridici o incide in modo
analogamente significativo e che la persona ritiene negativa per salute, sicurezza o
diritti fondamentali, ha diritto di ottenere spiegazioni chiare e significative sul
ruolo del sistema di IA nel processo decisionale e sui principali elementi della
decisione. Il diritto si applica nella misura in cui non sia già previsto da altre
norme dell'Unione.

## [GDPR, art. 22] Decisioni basate unicamente su trattamento automatizzato

Una persona ha diritto a non essere sottoposta a una decisione basata unicamente su un
trattamento automatizzato che produca effetti giuridici o incida significativamente su
di lei, salvo eccezioni (contratto, legge, consenso esplicito). Anche nelle eccezioni
devono esserci garanzie: almeno il diritto di ottenere l'intervento umano, di
esprimere la propria opinione e di contestare la decisione. Gli artt. 13-15 GDPR
prevedono informazioni significative sulla logica utilizzata.

## [Direttiva (UE) 2023/2225, art. 18] Valutazione del merito creditizio

La nuova direttiva sul credito ai consumatori richiede che la valutazione del merito
creditizio si basi su informazioni pertinenti e accurate sulla situazione economica
del consumatore. Se la valutazione comporta un trattamento automatizzato, il
consumatore ha diritto di chiedere e ottenere l'intervento umano, una spiegazione
chiara e comprensibile della valutazione e della logica e dei rischi del trattamento,
e di esprimere il proprio punto di vista e chiedere un riesame.

## [Buone pratiche] Cosa rende una spiegazione utile

Una spiegazione utile al cliente indica l'esito, i principali fattori che lo hanno
determinato con la loro direzione, cosa potrebbe cambiare l'esito (controfattuali
realistici e attuabili) e come chiedere un riesame umano. Deve essere fedele al
modello: citare fattori che il modello non ha usato, o invertirne l'effetto, rende la
spiegazione fuorviante anche se suona plausibile.
