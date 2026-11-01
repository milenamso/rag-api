"""Question pairs for the semantic cache experiment.

PARAPHRASES: same intent, different wording -> the cache SHOULD reuse the answer.
DIFFERENT: lexically close but asking something else -> the cache must NOT reuse it.
Each pair exists in Portuguese and English with the same meaning.
"""

Pair = dict[str, tuple[str, str]]

PARAPHRASES: list[Pair] = [
    {
        "pt": ("Como instalo o projeto?", "Como faço para instalar o projeto?"),
        "en": ("How do I install the project?", "What are the steps to install the project?"),
    },
    {
        "pt": ("Qual é o prazo de reembolso?", "Em quantos dias consigo o reembolso?"),
        "en": ("What is the refund deadline?", "How many days do I have to get a refund?"),
    },
    {
        "pt": ("Como redefinir minha senha?", "Esqueci minha senha, como troco?"),
        "en": ("How do I reset my password?", "I forgot my password, how can I change it?"),
    },
    {
        "pt": ("Quais formatos de arquivo são aceitos?", "Que tipos de arquivo posso enviar?"),
        "en": ("Which file formats are supported?", "What kinds of files can I upload?"),
    },
    {
        "pt": ("Como faço o deploy da aplicação?", "Como publico a aplicação em produção?"),
        "en": ("How do I deploy the application?", "How do I ship the app to production?"),
    },
    {
        "pt": ("Qual é o limite de tamanho do upload?", "Qual o tamanho máximo de arquivo?"),
        "en": ("What is the upload size limit?", "What is the maximum file size?"),
    },
    {
        "pt": ("Como cancelo minha assinatura?", "Quero cancelar o meu plano, como faço?"),
        "en": ("How do I cancel my subscription?", "I want to cancel my plan, how?"),
    },
    {
        "pt": ("Quem pode acessar os relatórios?", "Quais usuários têm acesso aos relatórios?"),
        "en": ("Who can access the reports?", "Which users have access to the reports?"),
    },
    {
        "pt": ("O sistema funciona offline?", "Dá para usar o sistema sem internet?"),
        "en": ("Does the system work offline?", "Can I use the system without internet?"),
    },
    {
        "pt": ("Como configuro a chave da API?", "Onde coloco a minha API key?"),
        "en": ("How do I configure the API key?", "Where do I put my API key?"),
    },
    {
        "pt": ("Quanto custa o plano empresarial?", "Qual o preço do plano para empresas?"),
        "en": ("How much is the enterprise plan?", "What is the price of the business plan?"),
    },
    {
        "pt": ("Como exporto os dados para CSV?", "Consigo baixar os dados em CSV?"),
        "en": ("How do I export data to CSV?", "Can I download the data as CSV?"),
    },
    {
        "pt": ("Qual é o horário do suporte?", "Em que horário o suporte atende?"),
        "en": ("What are the support hours?", "When is support available?"),
    },
    {
        "pt": ("Como adiciono um novo usuário?", "Como cadastro outra pessoa na conta?"),
        "en": ("How do I add a new user?", "How do I invite someone else to the account?"),
    },
    {
        "pt": (
            "Os dados são criptografados?",
            "As informações ficam armazenadas com criptografia?",
        ),
        "en": ("Is the data encrypted?", "Is information stored with encryption?"),
    },
    {
        "pt": ("Como atualizo para a versão mais nova?", "Como faço o upgrade da versão?"),
        "en": ("How do I update to the latest version?", "How do I upgrade the version?"),
    },
    {
        "pt": ("Onde vejo os logs de erro?", "Como acesso os logs com os erros?"),
        "en": ("Where can I see the error logs?", "How do I access the logs with errors?"),
    },
    {
        "pt": ("Qual banco de dados o projeto usa?", "Que banco de dados é utilizado?"),
        "en": ("Which database does the project use?", "What database is used?"),
    },
    {
        "pt": ("Como rodo os testes?", "Qual comando executa a suíte de testes?"),
        "en": ("How do I run the tests?", "Which command runs the test suite?"),
    },
    {
        "pt": ("Posso usar no Windows?", "O projeto é compatível com Windows?"),
        "en": ("Can I use it on Windows?", "Is the project compatible with Windows?"),
    },
]

DIFFERENT: list[Pair] = [
    {
        "pt": ("Como instalo o projeto no Windows?", "Como instalo o projeto no Linux?"),
        "en": ("How do I install the project on Windows?", "How do I install it on Linux?"),
    },
    {
        "pt": ("Qual é o prazo de reembolso?", "Qual é o prazo de entrega?"),
        "en": ("What is the refund deadline?", "What is the delivery deadline?"),
    },
    {
        "pt": ("Como redefinir minha senha?", "Como redefinir meu e-mail?"),
        "en": ("How do I reset my password?", "How do I reset my email address?"),
    },
    {
        "pt": ("Quais formatos de arquivo são aceitos?", "Quais formas de pagamento são aceitas?"),
        "en": ("Which file formats are supported?", "Which payment methods are supported?"),
    },
    {
        "pt": ("Como faço o deploy da aplicação?", "Como faço o rollback da aplicação?"),
        "en": ("How do I deploy the application?", "How do I roll back the application?"),
    },
    {
        "pt": (
            "Qual é o limite de tamanho do upload?",
            "Qual é o limite de requisições por minuto?",
        ),
        "en": ("What is the upload size limit?", "What is the request rate limit per minute?"),
    },
    {
        "pt": ("Como cancelo minha assinatura?", "Como renovo minha assinatura?"),
        "en": ("How do I cancel my subscription?", "How do I renew my subscription?"),
    },
    {
        "pt": ("Quem pode acessar os relatórios?", "Quem pode apagar os relatórios?"),
        "en": ("Who can access the reports?", "Who can delete the reports?"),
    },
    {
        "pt": ("O sistema funciona offline?", "O sistema funciona no celular?"),
        "en": ("Does the system work offline?", "Does the system work on mobile?"),
    },
    {
        "pt": ("Como configuro a chave da API?", "Como revogo a chave da API?"),
        "en": ("How do I configure the API key?", "How do I revoke the API key?"),
    },
    {
        "pt": ("Quanto custa o plano empresarial?", "Quanto custa o plano gratuito?"),
        "en": ("How much is the enterprise plan?", "How much is the free plan?"),
    },
    {
        "pt": ("Como exporto os dados para CSV?", "Como importo os dados de um CSV?"),
        "en": ("How do I export data to CSV?", "How do I import data from a CSV?"),
    },
    {
        "pt": ("Qual é o horário do suporte?", "Qual é o telefone do suporte?"),
        "en": ("What are the support hours?", "What is the support phone number?"),
    },
    {
        "pt": ("Como adiciono um novo usuário?", "Como removo um usuário?"),
        "en": ("How do I add a new user?", "How do I remove a user?"),
    },
    {
        "pt": ("Os dados são criptografados?", "Os dados são apagados após 30 dias?"),
        "en": ("Is the data encrypted?", "Is the data deleted after 30 days?"),
    },
    {
        "pt": ("Como atualizo para a versão mais nova?", "Como volto para a versão anterior?"),
        "en": (
            "How do I update to the latest version?",
            "How do I go back to the previous version?",
        ),
    },
    {
        "pt": ("Onde vejo os logs de erro?", "Onde vejo as métricas de uso?"),
        "en": ("Where can I see the error logs?", "Where can I see the usage metrics?"),
    },
    {
        "pt": ("Qual banco de dados o projeto usa?", "Qual linguagem o projeto usa?"),
        "en": ("Which database does the project use?", "Which language does the project use?"),
    },
    {
        "pt": ("Como rodo os testes?", "Como rodo o linter?"),
        "en": ("How do I run the tests?", "How do I run the linter?"),
    },
    {
        "pt": ("Posso usar no Windows?", "Posso usar comercialmente?"),
        "en": ("Can I use it on Windows?", "Can I use it commercially?"),
    },
]
