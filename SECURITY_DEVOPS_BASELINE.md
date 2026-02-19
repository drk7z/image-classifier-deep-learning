# Security & DevOps Baseline (Python/Gradio)

Este projeto é **Python + Gradio** (não Node.js). Este guia adapta seu kit para o stack atual.

## ✅ O que foi implementado

- Hardening de upload em `app.py`:
  - Limite de tamanho para imagem (`10 MB`)
  - Limite de tamanho para modelo (`200 MB`)
  - Validação de MIME (`image/*`)
  - Verificação de imagem com Pillow (`verify()`)
  - Arquivo temporário seguro com `tempfile.NamedTemporaryFile`
- CI em `.github/workflows/ci.yml`:
  - `compileall` (sanidade de sintaxe)
  - `pip check` (conflitos de dependência)
  - `pip-audit` (vulnerabilidades conhecidas)
  - `pytest` (quando houver testes Python)
- Containerização em `Dockerfile`:
  - Base `python:3.11-slim`
  - Processo rodando como usuário não-root
  - Configuração segura para execução headless do Gradio
- `.dockerignore` para reduzir superfície e vazamento de artefatos.

## ⚠️ Gaps atuais (próximos passos)

- Adicionar autenticação/autorização se o app for público (hoje é app aberto de inferência).
- Adicionar controle de rate limiting no reverse proxy (Nginx/Cloudflare/Azure Front Door).
- Versionar dependências com pinning mais estrito e estratégia de update (ex.: Dependabot).
- Incluir observabilidade centralizada (logs estruturados + métricas + alertas).
- Definir estratégia de secrets para produção (GitHub Secrets/Azure Key Vault).

## Checklist rápido para produção

1. Publicar atrás de HTTPS + WAF.
2. Não expor endpoint sem limite de upload/rate limit.
3. Monitorar CVEs no pipeline.
4. Rodar com usuário não-root e imagem mínima.
5. Revisar permissões mínimas para infra e storage.
