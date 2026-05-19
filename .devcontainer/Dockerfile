# Codespaces-only extension of the published image. We can't modify the
# published :latest until publish-image starts succeeding again, so we
# layer PR #11's full sshd bootstrap on top at devcontainer build time.
# This is a verbatim copy of the relevant block in the repo-root Dockerfile.
FROM ghcr.io/facebookexperimental/socmate:latest

RUN BASH_BIN="$(command -v bash)" \
 && SSHD_BIN="$(command -v sshd)" \
 && SSH_KEYGEN_BIN="$(command -v ssh-keygen)" \
 && test -x "${BASH_BIN}" \
 && test -x "${SSHD_BIN}" \
 && test -x "${SSH_KEYGEN_BIN}" \
 && ln -sf "${BASH_BIN}" /bin/bash \
 && mkdir -p /usr/sbin /usr/bin \
 && ln -sf "${SSHD_BIN}" /usr/sbin/sshd \
 && ln -sf "${SSH_KEYGEN_BIN}" /usr/bin/ssh-keygen \
 && if ! getent passwd sshd >/dev/null 2>&1; then \
        if command -v useradd >/dev/null 2>&1; then \
            useradd -r -M -d /var/empty -s /usr/sbin/nologin sshd; \
        else \
            echo 'sshd:x:74:74:Privilege-separated SSH:/var/empty:/usr/sbin/nologin' >> /etc/passwd \
         && echo 'sshd:x:74:' >> /etc/group; \
        fi; \
    fi \
 && mkdir -p /etc/ssh /var/run/sshd /run/sshd /root/.ssh /var/empty \
 && chmod 700 /root/.ssh \
 && chmod 755 /var/empty \
 && ssh-keygen -A \
 && python3 -c "p='/etc/shadow'; t=open(p).read(); open(p,'w').write(t.replace('root:!:','root:*:',1))" \
 && SFTP_SERVER="$(find /nix/store -path '*openssh*/libexec/sftp-server' -type f 2>/dev/null | head -1)" \
 && test -x "${SFTP_SERVER}" \
 && { \
        echo "Port 22"; \
        echo "PermitRootLogin prohibit-password"; \
        echo "PasswordAuthentication no"; \
        echo "PubkeyAuthentication yes"; \
        echo "AuthorizedKeysFile /root/.ssh/authorized_keys"; \
        echo "ChallengeResponseAuthentication no"; \
        echo "UsePAM no"; \
        echo "PrintMotd no"; \
        echo "AcceptEnv LANG LC_*"; \
        echo "Subsystem sftp ${SFTP_SERVER}"; \
    } > /etc/ssh/sshd_config
